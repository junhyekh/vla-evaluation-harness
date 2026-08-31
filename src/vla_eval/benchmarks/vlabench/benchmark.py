"""VLABench benchmark implementation.

VLABench is a large-scale benchmark for language-conditioned robotics
manipulation with long-horizon reasoning tasks, built on dm_control (MuJoCo).

Actions from the model server are 7-D end-effector targets or deltas plus a
gripper command. They are converted to joint-space via inverse kinematics and
sent to the dm_control environment as ``[7D qpos, 2D gripper]``.

Success is detected via dm_control's ``timestep.last()`` termination signal.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.render import configure_mujoco_render
from vla_eval.specs import (
    GRIPPER_RAW,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_ABSOLUTE,
    POSITION_DELTA,
    RAW,
    ROTATION_EULER,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULT_TASKS = [
    "select_fruit",
    "select_toy",
    "select_drink",
    "select_book",
    "select_painting",
]

DEFAULT_MAX_STEPS = 200
XR1_CAMERA_INDICES = {"front": 2, "base": 0, "wrist": 3}


class VLABenchBenchmark(StepBenchmark):
    """VLABench manipulation benchmark (dm_control / MuJoCo).

    Args:
        tasks: List of VLABench task names to evaluate.
        robot: Robot name (default ``"franka"``).
        max_steps: Optional cap on the upstream per-task horizon.
        eval_track: Upstream deterministic evaluation track name.
        seed: Seed used only when ``eval_track`` is unset.
        absolute_action: Treat position and Euler values as absolute targets.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    render_backends = frozenset({"gpu", "cpu"})

    @classmethod
    def configure_render(cls, mode: str) -> dict[str, str]:
        return configure_mujoco_render(mode)

    def __init__(
        self,
        tasks: list[str] | None = None,
        robot: str = "franka",
        max_steps: int | None = None,
        eval_track: str | None = None,
        seed: int = 42,
        absolute_action: bool = False,
        gripper_threshold: float = 0.0,
        gripper_open_value: float = 0.04,
        intention_score_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        if gripper_open_value < 0:
            raise ValueError("gripper_open_value must be non-negative")
        self._episode_configs = self._load_eval_track(eval_track) if eval_track else None
        available = list(self._episode_configs) if self._episode_configs else DEFAULT_TASKS
        self._task_names = list(tasks) if tasks is not None else available
        if self._episode_configs:
            unknown = sorted(set(self._task_names) - set(available))
            if unknown:
                raise ValueError(f"tasks are not present in {eval_track}: {unknown}")
        self._robot = robot
        self._max_steps = max_steps
        self._eval_track = eval_track
        self._seed = seed
        self._absolute_action = absolute_action
        self._gripper_threshold = gripper_threshold
        self._gripper_open_value = gripper_open_value
        self._intention_score_threshold = intention_score_threshold
        self._env: Any = None
        self._current_task: str | None = None
        self._instruction: str = ""
        self._last_ee_state: np.ndarray | None = None
        self._steps = 0
        self._horizon = DEFAULT_MAX_STEPS

    @staticmethod
    def _load_eval_track(eval_track: str) -> dict[str, list[dict[str, Any]]]:
        if Path(eval_track).name != eval_track:
            raise ValueError("eval_track must be a track name, not a path")
        root = os.environ.get("VLABENCH_ROOT")
        if not root:
            raise EnvironmentError("VLABENCH_ROOT is required when eval_track is set")
        path = Path(root) / "configs" / "evaluation" / "tracks" / f"{eval_track}.json"
        if not path.is_file():
            raise FileNotFoundError(f"VLABench evaluation track not found: {path}")
        data = json.loads(path.read_text())
        if (
            not isinstance(data, dict)
            or not data
            or any(not isinstance(value, list) or not value for value in data.values())
        ):
            raise ValueError(f"VLABench evaluation track is empty or malformed: {path}")
        return data

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def _ensure_vlabench(self) -> None:
        """Lazy-import VLABench and register robots/tasks."""
        import VLABench  # noqa: F401 — triggers VLABENCH_ROOT
        import VLABench.robots  # noqa: F401 — registers robot classes
        import VLABench.tasks  # noqa: F401 — registers task classes

        # Monkey-patch to skip PCD generator (Open3D segfaults in headless
        # containers and we never request point clouds).
        from VLABench.envs.dm_env import LM4ManipDMEnv

        if not hasattr(LM4ManipDMEnv, "_pcd_patched"):
            # Stub that satisfies `self.pcd_generator.physics = ...`
            class _PcdStub:
                physics = None

            LM4ManipDMEnv.register_pcd_generator = lambda self: setattr(self, "pcd_generator", _PcdStub())
            LM4ManipDMEnv._pcd_patched = True

    def get_tasks(self) -> list[Task]:
        return [{"name": t} for t in self._task_names]

    def _task_horizon(self, task_name: str) -> int:
        from VLABench.configs import name2config
        from VLABench.envs import TASK_CONFIG
        from VLABench.utils.utils import find_key_by_value

        series = find_key_by_value(name2config, task_name)
        upstream = TASK_CONFIG.get(series, {}).get("evaluation", {}).get("max_episode_length", DEFAULT_MAX_STEPS)
        return min(int(upstream), self._max_steps) if self._max_steps is not None else int(upstream)

    def reset(self, task: Task) -> Any:
        self._ensure_vlabench()
        from VLABench.envs import load_env

        task_name = task["name"]
        episode_idx = int(task.get("episode_idx", 0))
        episode_config = None
        if self._episode_configs is not None:
            episodes = self._episode_configs[task_name]
            if episode_idx >= len(episodes):
                raise IndexError(f"{self._eval_track}/{task_name} has only {len(episodes)} episodes")
            episode_config = episodes[episode_idx]
        else:
            np.random.seed(self._seed + episode_idx)
            random.seed(self._seed + episode_idx)

        # Close previous env and create new one (task may change scene layout)
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning("Failed to close VLABench environment: %s", e)
        self._env = load_env(
            task_name,
            robot=self._robot,
            episode_config=episode_config,
            random_init=episode_config is None,
            eval=False,
            run_mode="eval",
        )
        self._env.reset()
        self._current_task = task_name
        self._steps = 0
        self._horizon = self._task_horizon(task_name)

        obs = self._env.get_observation(require_pcd=False)
        self._instruction = self._env.task.get_instruction()
        self._last_ee_state = obs.get("ee_state", None)
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: Action) -> StepResult:
        from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raise ValueError("VLABench requires an 'actions' vector")
        raw_action = np.asarray(raw_action, dtype=np.float64)
        if raw_action.shape != (7,):
            raise ValueError(f"VLABench expected a 7-D action, got {raw_action.shape}")

        # Interpret 7D action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        delta_pos = raw_action[:3]
        delta_euler = raw_action[3:6]
        gripper_cmd = raw_action[6] if len(raw_action) > 6 else 0.0

        if self._absolute_action:
            target_pos, target_euler = delta_pos, delta_euler
        else:
            ee_state = self._last_ee_state
            if ee_state is None:
                ee_state = np.concatenate([self._env.get_ee_pos(), self._env.get_ee_quat(), [0.0]])
            target_pos = ee_state[:3] + delta_pos
            target_euler = np.array(quaternion_to_euler(ee_state[3:7])) + delta_euler
        target_quat = euler_to_quaternion(*target_euler)

        # Inverse kinematics: ee pose → joint positions
        _, qpos = self._env.robot.get_qpos_from_ee_pos(
            physics=self._env.physics,
            pos=target_pos,
            quat=target_quat,
        )

        # Gripper threshold and travel are policy/robot calibration knobs.
        grip_val = self._gripper_open_value if gripper_cmd >= self._gripper_threshold else 0.0
        gripper_state = np.array([grip_val, grip_val])
        env_action = np.concatenate([qpos, gripper_state])

        timestep = self._env.step(env_action)
        success = bool(timestep.last())
        self._steps += 1
        done = success or self._steps >= self._horizon

        # Update cached EE state
        obs = self._env.get_observation(require_pcd=False)
        self._last_ee_state = obs.get("ee_state", None)
        self._instruction = self._env.task.get_instruction()

        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=1.0 if success else 0.0, done=done, success=success)

        return StepResult(
            obs=obs,
            reward=1.0 if success else 0.0,
            done=done,
            info={"success": success, "timestep": timestep},
        )

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, dict):
            return None
        rgb = raw_obs.get("rgb")
        if rgb is None or len(rgb) == 0:
            return None
        return np.asarray(rgb[0])

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        if not isinstance(raw_obs, dict):
            raise TypeError("VLABench observation must be a mapping")
        rgb = np.asarray(raw_obs.get("rgb"))
        if rgb.ndim != 4 or rgb.shape[0] < 4 or rgb.shape[-1] != 3:
            raise ValueError(f"VLABench expected at least four RGB HWC views, got {rgb.shape}")
        ee_state = np.asarray(raw_obs.get("ee_state"), dtype=np.float32).reshape(-1)
        if ee_state.shape != (8,):
            raise ValueError(f"VLABench expected an 8-D ee_state, got {ee_state.shape}")

        return {
            "images": {name: np.ascontiguousarray(rgb[index]) for name, index in XR1_CAMERA_INDICES.items()},
            "state": ee_state,
            "task_description": self._instruction,
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {
            "success": bool(step_result.info.get("success", False)),
            "intention_score": float(self._env.get_intention_score(threshold=self._intention_score_threshold)),
            "progress_score": float(self._env.get_task_progress()),
        }

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "intention_score": "mean", "progress_score": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "max_steps": self._max_steps or 300,
            "eval_track": self._eval_track,
            "seed": self._seed,
        }
        if self._episode_configs:
            metadata["max_episodes_per_task"] = min(len(self._episode_configs[name]) for name in self._task_names)
        return metadata

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_ABSOLUTE if self._absolute_action else POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}
