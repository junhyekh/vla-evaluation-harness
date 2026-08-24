"""UR5e + Robotiq85 cross-embodiment contract."""

from pathlib import Path

import numpy as np

from vla_eval.benchmarks.libero_ur5.benchmark import _panda_like_gripper_state, _transcode_panda_state
from vla_eval.cli.config_loader import load_config

ROOT = Path(__file__).parents[1]


def test_transcode_preserves_object_state_and_target_robot() -> None:
    # Two free object joints: 14 qpos and 12 qvel. Panda has 9+9 robot state;
    # UR5e + Robotiq85 has 12+12.
    source = np.arange(1 + 23 + 21, dtype=np.float64)
    target = np.full(1 + 26 + 24, -1.0)
    result, robot_nq, robot_nv = _transcode_panda_state(source, target, 26, 24)

    assert (robot_nq, robot_nv) == (12, 12)
    np.testing.assert_array_equal(result[13:27], source[10:24])
    np.testing.assert_array_equal(result[39:], source[33:])
    np.testing.assert_array_equal(result[1:13], target[1:13])


def test_robotiq_state_maps_to_panda_finger_pair() -> None:
    np.testing.assert_allclose(_panda_like_gripper_state(np.zeros(6)), [0.04, -0.04])
    np.testing.assert_allclose(
        _panda_like_gripper_state(np.array([0.8, 0, 0, 0.8, 0, 0])),
        [0.0, 0.0],
    )


def test_libero_ur5_object_orders() -> None:
    normal = load_config(str(ROOT / "configs/benchmarks/libero_ur5/object.yaml"))["benchmarks"]
    assert len(normal) == 1
    assert normal[0]["episodes_per_task"] == 50
    assert normal[0]["params"]["suite"] == "libero_object"

    pro = load_config(str(ROOT / "configs/benchmarks/libero_pro_ur5/object.yaml"))["benchmarks"]
    assert [entry["params"]["suite"] for entry in pro] == [
        "libero_object_object",
        "libero_object_swap",
        "libero_object_lan",
        "libero_object_task",
    ]
    assert all(entry["episodes_per_task"] == 50 for entry in pro)
    assert all(entry["params"]["max_steps"] == 280 for entry in pro)
