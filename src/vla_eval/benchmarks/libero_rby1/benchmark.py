"""LIBERO tasks with a fixed RBY1-A body and controlled right arm."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.benchmarks.libero_rby1.robot import register_rby1_right


class LIBERORBY1Benchmark(LIBEROBenchmark):
    """Cross-embodiment LIBERO; scores are not official Panda LIBERO scores."""

    def __init__(
        self,
        suite: str = "libero_object",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
        max_steps: int | None = None,
        env_seed: int | None = None,
        model_dir: str | None = None,
    ) -> None:
        model_dir = model_dir or os.environ.get(
            "VLA_EVAL_RBY1_MODEL_DIR",
            "/app/mujoco_menagerie/rainbow_robotics_rby1",
        )
        register_rby1_right(model_dir)
        super().__init__(
            suite=suite,
            seed=seed,
            num_steps_wait=num_steps_wait,
            send_wrist_image=send_wrist_image,
            send_state=send_state,
            max_steps=max_steps,
            env_seed=env_seed,
            robot="RBY1Right",
        )

    def _set_init_state(self, initial_state: np.ndarray) -> Any:
        """Keep the official object state while replacing Panda joint state."""
        assert self._env is not None
        env = self._env
        state = np.asarray(initial_state, dtype=np.float64).copy()
        expected = 1 + env.sim.model.nq + env.sim.model.nv
        if state.size != expected:
            raise ValueError(f"RBY1/Panda state layouts differ: got {state.size}, expected {expected}")

        robot = env.robots[0]
        # Panda's OSC gains under-drive the heavier RBY arm.
        robot.controller.kp[:] = 300.0
        robot.controller.kd[:] = 2.0 * np.sqrt(robot.controller.kp)
        for index, value in zip(robot._ref_joint_pos_indexes, robot.init_qpos):
            state[1 + index] = value
        for index, value in zip(robot._ref_gripper_joint_pos_indexes, robot.gripper.init_qpos):
            state[1 + index] = value

        velocity_offset = 1 + env.sim.model.nq
        state[velocity_offset + np.asarray(robot._ref_joint_vel_indexes)] = 0.0
        state[velocity_offset + np.asarray(robot._ref_gripper_joint_vel_indexes)] = 0.0
        return env.regenerate_obs_from_state(state)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(embodiment="rby1_a_fixed_right", official_libero_score=False)
        return metadata
