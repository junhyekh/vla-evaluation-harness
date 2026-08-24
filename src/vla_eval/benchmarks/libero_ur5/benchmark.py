"""LIBERO tasks with robosuite's native UR5e and Robotiq85 gripper."""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.benchmarks.libero_ur5.robot import register_ur5e
from vla_eval.types import Observation, Task

_PANDA_ROBOT_NQ = 9
_PANDA_ROBOT_NV = 9


def _transcode_panda_state(
    initial_state: np.ndarray,
    target_state: np.ndarray,
    target_nq: int,
    target_nv: int,
) -> tuple[np.ndarray, int, int]:
    """Copy official object state around a different robot joint layout."""
    source = np.asarray(initial_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64).copy()
    if target.size != 1 + target_nq + target_nv:
        raise ValueError(f"Bad UR5 state size: got {target.size}, expected {1 + target_nq + target_nv}")

    # Both models contain the same free object joints, so nq - nv is invariant.
    source_sum = source.size - 1
    source_delta = target_nq - target_nv
    if (source_sum + source_delta) % 2:
        raise ValueError(f"Cannot infer Panda state layout from {source.size} values")
    source_nq = (source_sum + source_delta) // 2
    source_nv = source_sum - source_nq

    object_nq = source_nq - _PANDA_ROBOT_NQ
    object_nv = source_nv - _PANDA_ROBOT_NV
    target_robot_nq = target_nq - object_nq
    target_robot_nv = target_nv - object_nv
    if min(object_nq, object_nv, target_robot_nq, target_robot_nv) < 0:
        raise ValueError("Panda and UR5 object state layouts are incompatible")

    target[0] = source[0]
    target[1 + target_robot_nq : 1 + target_nq] = source[1 + _PANDA_ROBOT_NQ : 1 + source_nq]
    target[1 + target_nq + target_robot_nv :] = source[1 + source_nq + _PANDA_ROBOT_NV :]
    return target, target_robot_nq, target_robot_nv


def _panda_like_gripper_state(gripper_qpos: np.ndarray) -> np.ndarray:
    """Map Robotiq85 closure to the 2-D Panda state expected by pi0.5-LIBERO."""
    qpos = np.asarray(gripper_qpos, dtype=np.float32)
    if qpos.shape != (6,):
        raise ValueError(f"Expected six Robotiq85 joints, got {qpos.shape}")
    closure = float(np.clip(np.mean(qpos[[0, 3]]) / 0.8, 0.0, 1.0))
    opening = 0.04 * (1.0 - closure)
    return np.array([opening, -opening], dtype=np.float32)


class LIBEROUR5Benchmark(LIBEROBenchmark):
    """Non-official UR5e + Robotiq85 cross-embodiment LIBERO evaluation."""

    def __init__(
        self,
        suite: str = "libero_object",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
        max_steps: int | None = None,
        env_seed: int | None = None,
    ) -> None:
        register_ur5e()
        super().__init__(
            suite=suite,
            seed=seed,
            num_steps_wait=num_steps_wait,
            send_wrist_image=send_wrist_image,
            send_state=send_state,
            max_steps=max_steps,
            env_seed=env_seed,
            robot="UR5e",
        )

    def _set_init_state(self, initial_state: np.ndarray) -> Any:
        assert self._env is not None
        env = self._env
        state, robot_nq, robot_nv = _transcode_panda_state(
            initial_state,
            env.sim.get_state().flatten(),
            env.sim.model.nq,
            env.sim.model.nv,
        )
        robot = env.robots[0]
        if max(robot._ref_gripper_joint_pos_indexes) >= robot_nq:
            raise ValueError("UR5 robot qpos are not the leading simulator state")
        for index, value in zip(robot._ref_joint_pos_indexes, robot.init_qpos):
            state[1 + index] = value
        for index, value in zip(robot._ref_gripper_joint_pos_indexes, robot.gripper.init_qpos):
            state[1 + index] = value

        velocity_offset = 1 + env.sim.model.nq
        state[velocity_offset : velocity_offset + robot_nv] = 0.0
        return env.regenerate_obs_from_state(state)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        # pi0.5-LIBERO was normalized on Panda's two finger positions. Preserve
        # that 8-D state contract instead of silently sending six Robotiq joints.
        if self.send_state:
            raw_obs = dict(raw_obs)
            raw_obs["robot0_gripper_qpos"] = _panda_like_gripper_state(raw_obs["robot0_gripper_qpos"])
        return super().make_obs(raw_obs, task)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            embodiment="ur5e_robotiq85",
            gripper="Robotiq85Gripper",
            gripper_speed=0.2,
            official_libero_score=False,
        )
        return metadata
