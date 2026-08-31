"""XR-1 processor contract checks that do not require the checkpoint or CUDA."""

import json
from types import SimpleNamespace

import numpy as np

from vla_eval.benchmarks.vlabench.benchmark import VLABenchBenchmark
from vla_eval.model_servers.xr1 import (
    ROBOCASA,
    ROBOCASA365,
    VLABENCH,
    XR1ModelServer,
    _build_messages,
    _padded_state,
    _robocasa365_state,
    _sample_history,
    _vlabench_state,
    _vlabench_targets,
)
from vla_eval.specs import check_specs


def test_profile_preprocessing_contracts() -> None:
    assert _sample_history(list(range(7))) == [0, 2, 4, 6]
    assert _sample_history([9]) == [9, 9, 9, 9]

    robocasa365 = _robocasa365_state(
        {
            "state.end_effector_position_relative": [1, 2, 3],
            "state.end_effector_rotation_relative": [0, 0, 0, 1],
            "state.gripper_qpos": [4, 5],
            "state.base_position": [6, 7, 8],
            "state.base_rotation": [0, 0, 0, 1],
        }
    )
    np.testing.assert_allclose(robocasa365, [1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 0, 0, 0])
    assert _padded_state([robocasa365]).shape == (1, 1, 60)

    vlabench = _vlabench_state([0, -0.4, 0.78, 1, 0, 0, 0, 0.5])
    np.testing.assert_allclose(vlabench, [0, 0, 0, 0, 0, 0, 0.5], atol=1e-6)

    targets = _vlabench_targets(
        np.array([[0.1, 0, 0, 0, 0, 0.2, 0.3], [0, 0.2, 0, 0, 0.1, 0, 0.1]]),
        [1, 2, 3, 1, 0, 0, 0, 0.5],
    )
    np.testing.assert_allclose(
        targets,
        [[1.1, 2, 3, 0, 0, 0.2, 0.3], [1.1, 2.2, 3, 0, 0.1, 0.2, 0.1]],
        atol=1e-6,
    )


def test_profile_prompts_preserve_camera_order() -> None:
    for robot_type, media_type in ((ROBOCASA, "image"), (ROBOCASA365, "video"), (VLABENCH, "image")):
        messages = _build_messages(robot_type, ["first", "second", "third"], "do the task")
        media = [item for item in messages[0]["content"] if item["type"] == media_type]
        key = media_type
        assert [item[key] for item in media] == ["first", "second", "third"]
        separator = "\n\n" if robot_type == ROBOCASA365 else "\n"
        assert messages[0]["content"][-1]["text"] == (
            f"{separator}Generate robot actions for the task:\ndo the task /no_cot"
        )
        assert messages[1]["content"][0]["text"] == "<cot></cot>"


def test_vlabench_observation_connects_to_xr1_processor() -> None:
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return {}

    server = object.__new__(XR1ModelServer)
    server.robot_type = VLABENCH
    server.request_seed = 42
    server._processor = Processor()
    benchmark = VLABenchBenchmark(tasks=["select_toy"], **server.get_observation_params())
    benchmark._instruction = "pick the toy"

    assert (
        check_specs(
            server.get_action_spec(),
            benchmark.get_action_spec(),
            server.get_observation_spec(),
            benchmark.get_observation_spec(),
        )
        == []
    )

    cameras = np.stack([np.full((4, 4, 3), value, dtype=np.uint8) for value in (0, 40, 80, 120)])
    obs = benchmark.make_obs(
        {"rgb": cameras, "ee_state": np.array([0, -0.4, 0.78, 1, 0, 0, 0, 0.5])},
        {"name": "select_toy"},
    )
    payload = server._model_inputs(obs, SimpleNamespace(session_id="test"))

    media = [item["image"] for item in server._processor.messages[0]["content"] if item["type"] == "image"]
    assert [image.getpixel((0, 0))[0] for image in media] == [80, 0, 120]
    np.testing.assert_allclose(server._processor.kwargs["state"][0, 0, :7], [0, 0, 0, 0, 0, 0, 0.5])
    assert payload == {"task_id": VLABENCH, "seed": 42}


def test_vlabench_track_manifest_selects_fixed_episodes(tmp_path, monkeypatch) -> None:
    track_dir = tmp_path / "configs" / "evaluation" / "tracks"
    track_dir.mkdir(parents=True)
    (track_dir / "track_test.json").write_text(
        json.dumps({"select_toy": [{"episode": 0}, {"episode": 1}], "select_fruit": [{"episode": 0}]})
    )
    monkeypatch.setenv("VLABENCH_ROOT", str(tmp_path))

    benchmark = VLABenchBenchmark(tasks=["select_toy"], eval_track="track_test")

    assert benchmark.get_tasks() == [{"name": "select_toy"}]
    assert benchmark.get_metadata()["max_episodes_per_task"] == 2
