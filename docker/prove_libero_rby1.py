"""Recorded oracle proof for one LIBERO-Object task on RBY1-right."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vla_eval.benchmarks.video import EpisodeVideoRecorder
from vla_eval.benchmarks.libero_rby1.benchmark import LIBERORBY1Benchmark
from vla_eval.rotation import matrix_to_quat, quat_to_axisangle


def _assert_gripper_model(benchmark: LIBERORBY1Benchmark) -> None:
    assert benchmark._env is not None
    model = benchmark._env.sim.model
    geom_names = [
        name
        for name in model.geom_names
        if name and name.startswith("gripper0_") and ("hand_" in name or "finger_" in name)
    ]
    assert geom_names
    # mjGEOM_MESH == 7. This catches inherited robosuite sphere defaults.
    assert all(int(model.geom_type[model.geom_name2id(name)]) == 7 for name in geom_names)
    finger_r2 = model.body_name2id("gripper0_finger_r2")
    np.testing.assert_allclose(model.body_quat[finger_r2], [0.0, 0.0, 0.0, 1.0])


def _move(
    benchmark: LIBERORBY1Benchmark,
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
    benchmark = LIBERORBY1Benchmark(num_steps_wait=10)
    task = benchmark.get_tasks()[2]
    task["episode_idx"] = 0
    video = None
    recorder = None
    if video_dir:
        video = EpisodeVideoRecorder(
            output_dir=video_dir,
            filename="rby1_right_proof_{status}.mp4",
            fps=20,
            overwrite=True,
        )
        video.start({})
        recorder = SimpleNamespace(record_video=video.record, record_step=lambda **_: None)
    asyncio.run(benchmark.start_episode(task, recorder=recorder))
    assert benchmark._env is not None
    _assert_gripper_model(benchmark)

    env = benchmark._env.env
    sim = env.sim

    def body_pos(name: str) -> np.ndarray:
        return sim.data.body_xpos[sim.model.body_name2id(name)].copy()

    target_name, basket_name = env.obj_of_interest
    target_body = f"{target_name}_main"
    basket_body = f"{basket_name}_main"
    stages = []

    def record_stage(name: str) -> None:
        stages.append(
            {
                "name": name,
                "eef": np.round(benchmark._env.robots[0].controller.ee_pos, 3).tolist(),
                "eef_z": np.round(benchmark._env.robots[0].controller.ee_ori_mat[:, 2], 3).tolist(),
                "object": np.round(body_pos(target_body), 3).tolist(),
                "gripper_qpos": np.round(
                    sim.data.qpos[benchmark._env.robots[0]._ref_gripper_joint_pos_indexes], 3
                ).tolist(),
            }
        )

    object_pos = body_pos(target_body)
    vertical_grip = np.eye(3)
    record_stage("reset")
    actions = _move(benchmark, object_pos + [0.0, 0.0, 0.30], -1.0, 30, vertical_grip)
    record_stage("above_object")
    grasp_pos = object_pos + [0.025, -0.025, 0.02]
    actions += _move(benchmark, grasp_pos, -1.0, 40, vertical_grip)
    record_stage("grasp_open")
    actions += _move(benchmark, grasp_pos, 1.0, 10, vertical_grip)
    record_stage("grasp_closed")
    actions += _move(benchmark, object_pos + [0.0, 0.0, 0.35], 1.0, 20, vertical_grip)
    record_stage("lifted")

    basket_pos = body_pos(basket_body)
    above_basket = basket_pos + [0.0, 0.0, 0.35]
    actions += _move(benchmark, above_basket, 1.0, 120, vertical_grip)
    record_stage("above_basket")
    for _ in range(60):
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
        "eef": np.round(benchmark._env.robots[0].controller.ee_pos, 3).tolist(),
        "object": np.round(body_pos(target_body), 3).tolist(),
        "basket": np.round(body_pos(basket_body), 3).tolist(),
        "target": np.round(above_basket, 3).tolist(),
        "arm_qpos": np.round(sim.data.qpos[benchmark._env.robots[0]._ref_joint_pos_indexes], 3).tolist(),
        "gripper_geometry": "mesh",
        "stages": stages,
        "video": str(video_path) if video_path else None,
    }
    if video_dir:
        record_path = Path(video_dir) / f"rby1_right_proof_{'success' if success else 'fail'}.json"
        result["record"] = str(record_path)
        record_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    benchmark.cleanup()
    return 0 if success and actions <= max_steps else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir")
    sys.exit(main(parser.parse_args().video_dir))
