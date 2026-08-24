"""Recorded controller and gripper proof for one UR5e LIBERO-Object task."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vla_eval.benchmarks.libero_ur5.benchmark import LIBEROUR5Benchmark
from vla_eval.benchmarks.video import EpisodeVideoRecorder
from vla_eval.rotation import matrix_to_quat, quat_to_axisangle


def _move(
    benchmark: LIBEROUR5Benchmark,
    target: np.ndarray,
    gripper: float,
    steps: int,
    orientation: np.ndarray,
) -> int:
    assert benchmark._env is not None
    for _ in range(steps):
        controller = benchmark._env.robots[0].controller
        scale = np.asarray(controller.output_max)
        delta = np.clip((target - controller.ee_pos) / scale[:3], -1.0, 1.0)
        error = orientation @ controller.ee_ori_mat.T
        rotation = quat_to_axisangle(matrix_to_quat(error))
        rotation = np.clip(rotation / scale[3:6], -1.0, 1.0)
        benchmark.step({"actions": [*delta, *rotation, gripper]})
    return steps


def main(video_dir: str | None = None) -> int:
    benchmark = LIBEROUR5Benchmark(num_steps_wait=10)
    task = benchmark.get_tasks()[2]
    task["episode_idx"] = 0
    video = None
    recorder = None
    if video_dir:
        video = EpisodeVideoRecorder(
            output_dir=video_dir,
            filename="ur5e_robotiq85_proof_{status}.mp4",
            fps=20,
            overwrite=True,
        )
        video.start({})
        recorder = SimpleNamespace(record_video=video.record, record_step=lambda **_: None)
    asyncio.run(benchmark.start_episode(task, recorder=recorder))
    assert benchmark._env is not None

    robot = benchmark._env.robots[0]
    assert type(robot.gripper).__name__ == "LIBERORobotiq85Gripper"
    assert robot.gripper.dof == 1 and robot.gripper.speed == 0.2
    assert len(robot._ref_gripper_joint_pos_indexes) == 6

    env = benchmark._env.env
    sim = env.sim

    def body_pos(name: str) -> np.ndarray:
        return sim.data.body_xpos[sim.model.body_name2id(name)].copy()

    def jaw_gap() -> float:
        left = sim.data.geom_xpos[sim.model.geom_name2id("gripper0_left_fingerpad_collision")]
        right = sim.data.geom_xpos[sim.model.geom_name2id("gripper0_right_fingerpad_collision")]
        return float(np.linalg.norm(left - right))

    target_name, basket_name = env.obj_of_interest
    target_body = f"{target_name}_main"
    basket_body = f"{basket_name}_main"
    stages: list[dict[str, object]] = []

    def record_stage(name: str) -> None:
        stages.append(
            {
                "name": name,
                "eef": np.round(robot.controller.ee_pos, 3).tolist(),
                "eef_z": np.round(robot.controller.ee_ori_mat[:, 2], 3).tolist(),
                "object": np.round(body_pos(target_body), 3).tolist(),
                "jaw_gap_m": round(jaw_gap(), 4),
                "gripper_qpos": np.round(
                    sim.data.qpos[robot._ref_gripper_joint_pos_indexes], 3
                ).tolist(),
            }
        )

    object_pos = body_pos(target_body)
    # Robotiq's grip site points down when its local z axis is world -z.
    vertical_grip = np.diag([1.0, -1.0, -1.0])
    record_stage("reset")
    actions = _move(benchmark, object_pos + [0.0, 0.0, 0.30], -1.0, 30, vertical_grip)
    record_stage("above_object")
    grasp_pos = object_pos + [0.0, 0.0, 0.04]
    actions += _move(benchmark, grasp_pos, -1.0, 40, vertical_grip)
    record_stage("grasp_open")
    actions += _move(benchmark, grasp_pos, 1.0, 10, vertical_grip)
    record_stage("grasp_closed")
    actions += _move(benchmark, object_pos + [0.0, 0.0, 0.35], 1.0, 25, vertical_grip)
    record_stage("lifted")

    basket_pos = body_pos(basket_body)
    above_basket = basket_pos + [0.0, 0.0, 0.35]
    actions += _move(benchmark, above_basket, 1.0, 100, vertical_grip)
    record_stage("above_basket")
    for _ in range(50):
        actions += _move(benchmark, above_basket, -1.0, 1, vertical_grip)
        if env._check_success():
            break
    record_stage("released")

    success = bool(env._check_success())
    video_path = video.save(status="success" if success else "fail") if video else None
    max_steps = benchmark.get_metadata()["max_steps"]
    result = {
        "task": task["name"],
        "success": success,
        "actions": actions,
        "max_steps": max_steps,
        "robot": "UR5e",
        "gripper": "Robotiq85Gripper",
        "gripper_speed": robot.gripper.speed,
        "eef": np.round(robot.controller.ee_pos, 3).tolist(),
        "object": np.round(body_pos(target_body), 3).tolist(),
        "basket": np.round(body_pos(basket_body), 3).tolist(),
        "stages": stages,
        "video": str(video_path) if video_path else None,
    }
    if video_dir:
        record_path = Path(video_dir) / f"ur5e_robotiq85_proof_{'success' if success else 'fail'}.json"
        result["record"] = str(record_path)
        record_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    benchmark.cleanup()
    return 0 if success and actions <= max_steps else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir")
    sys.exit(main(parser.parse_args().video_dir))
