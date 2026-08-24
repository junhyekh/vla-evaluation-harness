"""LIBERO-Pro's released evaluation contract."""

from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.benchmarks.libero_pro.benchmark import LIBEROProBenchmark
from vla_eval.cli.config_loader import load_config


def test_libero_pro_released_order() -> None:
    benchmark = LIBEROProBenchmark(
        suite="libero_spatial",
        perturbation="language",
        max_steps=123,
        env_seed=0,
    )
    assert benchmark.suite == "libero_spatial_lan"
    assert benchmark.env_seed == 0
    assert benchmark.get_metadata() == {
        "max_steps": 123,
        "max_episodes_per_task": 50,
        "suite": "libero_spatial_lan",
        "base_suite": "libero_spatial",
        "perturbation": "lan",
    }

    config = load_config(
        str(Path(__file__).parents[1] / "configs/benchmarks/libero_pro/eval.yaml")
    )
    cells = {(entry["params"]["suite"], entry["params"]["perturbation"]) for entry in config["benchmarks"]}
    assert cells == set(
        product(
            ("libero_goal", "libero_spatial", "libero_10", "libero_object"),
            ("object", "swap", "lan", "task"),
        )
    )
    assert all(entry["episodes_per_task"] == 50 for entry in config["benchmarks"])


def test_libero_step_records_action_and_robot_state() -> None:
    obs = {
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.01, -0.01]),
    }
    benchmark = LIBEROBenchmark()
    benchmark._env = SimpleNamespace(step=lambda action: (obs, 0.0, False, {}))
    rows = []
    benchmark._recorder = SimpleNamespace(record_video=lambda frame: None, record_step=lambda **row: rows.append(row))

    benchmark.step({"actions": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.2])})

    assert rows == [
        {
            "reward": 0.0,
            "done": False,
            "success": False,
            "raw_action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.2],
            "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
            "eef_pos": obs["robot0_eef_pos"],
            "eef_quat": obs["robot0_eef_quat"],
            "gripper_qpos": obs["robot0_gripper_qpos"],
        }
    ]
