# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "flash-attn==2.8.3",
#     "ninja==1.13.0",
#     "numpy==2.1.3",
#     "pillow==11.3.0",
#     "torch==2.8.0",
#     "torchvision==0.23.0",
#     "transformers==4.57.1",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-08-30T00:00:00Z"
# no-build-isolation-package = ["flash-attn"]
# ///
"""Xiaomi-Robotics-1 model server for the released simulation checkpoints."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.rotation import matrix_to_euler_xyz, quat_to_axisangle, quat_to_matrix, quat_wxyz_to_xyzw
from vla_eval.specs import (
    BASE_MOTION,
    CONTROL_MODE_01,
    GRIPPER_CLOSE_01,
    GRIPPER_RAW,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_ABSOLUTE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    ROTATION_EULER,
    DimSpec,
)
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

ROBOCASA = "robocasa_mg"
ROBOCASA365 = "robocasa365"
VLABENCH = "vlabench_choice"
ROBOT_TYPES = (ROBOCASA, ROBOCASA365, VLABENCH)

ROBOCASA_CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
ROBOCASA365_CAMERAS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
VLABENCH_CAMERAS = ("front", "base", "wrist")

STATE_DIM = 60
ACTION_DIMS = {ROBOCASA: 7, ROBOCASA365: 12, VLABENCH: 7}
VLABENCH_POSITION_OFFSET = np.array([0.0, -0.4, 0.78], dtype=np.float32)
VLABENCH_GRIPPER_THRESHOLD = 0.2


def _normalized_quaternion(value: Any, *, order: str) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"expected a 4-D quaternion, got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quat = quat / norm
    return quat_wxyz_to_xyzw(quat) if order == "wxyz" else quat


def _robocasa365_state(state: Mapping[str, Any]) -> np.ndarray:
    """Pack Xiaomi's 14-D RoboCasa365 state from the harness's named fields."""
    required = (
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"RoboCasa365 state is missing fields: {missing}")
    packed = np.concatenate(
        [
            np.asarray(state[required[0]], dtype=np.float32).reshape(-1),
            quat_to_axisangle(_normalized_quaternion(state[required[1]], order="xyzw")),
            np.asarray(state[required[2]], dtype=np.float32).reshape(-1),
            np.asarray(state[required[3]], dtype=np.float32).reshape(-1),
            quat_to_axisangle(_normalized_quaternion(state[required[4]], order="xyzw")),
        ]
    ).astype(np.float32)
    if packed.shape != (14,):
        raise ValueError(f"expected a 14-D RoboCasa365 state, got {packed.shape}")
    return packed


def _vlabench_execution_state(state: Any) -> np.ndarray:
    ee_state = np.asarray(state, dtype=np.float32).reshape(-1)
    if ee_state.shape != (8,):
        raise ValueError(f"expected an 8-D VLABench end-effector state, got {ee_state.shape}")
    rotation = quat_to_matrix(_normalized_quaternion(ee_state[3:7], order="wxyz"))
    euler = matrix_to_euler_xyz(rotation)
    euler = (euler + np.pi) % (2.0 * np.pi) - np.pi
    return np.concatenate([ee_state[:3], euler, ee_state[7:8]]).astype(np.float32)


def _vlabench_state(state: Any) -> np.ndarray:
    """Convert VLABench ``[xyz, quaternion_wxyz, gripper]`` to XR-1's 7-D state."""
    result = _vlabench_execution_state(state)
    result[:3] -= VLABENCH_POSITION_OFFSET
    return result


def _vlabench_targets(deltas: np.ndarray, state: Any) -> np.ndarray:
    """Accumulate one XR-1 chunk into VLABench's absolute EE targets."""
    targets = np.asarray(deltas, dtype=np.float32).copy()
    if targets.ndim != 2 or targets.shape[1] != ACTION_DIMS[VLABENCH]:
        raise ValueError(f"expected VLABench action deltas shaped [T, 7], got {targets.shape}")
    current = _vlabench_execution_state(state)
    for target in targets:
        current[:6] += target[:6]
        current[3:6] = (current[3:6] + np.pi) % (2.0 * np.pi) - np.pi
        target[:6] = current[:6]
    return targets


def _padded_state(states: Sequence[np.ndarray]) -> np.ndarray:
    if not states:
        raise ValueError("state history cannot be empty")
    width = states[0].size
    if width > STATE_DIM or any(state.shape != (width,) for state in states):
        raise ValueError("state history has inconsistent or oversized entries")
    padded = np.zeros((1, len(states), STATE_DIM), dtype=np.float32)
    padded[0, :, :width] = np.stack(states)
    return padded


def _sample_history(items: Sequence[Any], length: int = 4, interval: int = 2) -> list[Any]:
    if not items:
        raise ValueError("observation history cannot be empty")
    indices = [max(0, len(items) - 1 - (length - 1 - index) * interval) for index in range(length)]
    return [items[index] for index in indices]


def _build_messages(robot_type: str, views: Sequence[Any], instruction: str) -> list[dict[str, Any]]:
    if len(views) != 3:
        raise ValueError(f"XR-1 expects three camera views, got {len(views)}")
    if robot_type == ROBOCASA:
        content = [
            {"type": "text", "text": "The following observations are captured from multiple views.\n# Base View\n"},
            {"type": "image", "image": views[0]},
            {"type": "image", "image": views[1]},
            {"type": "text", "text": "\n# Left-Wrist View\n"},
            {"type": "image", "image": views[2]},
        ]
    elif robot_type == ROBOCASA365:
        content = [
            {"type": "text", "text": "Left camera: "},
            {"type": "video", "video": views[0]},
            {"type": "text", "text": "\nRight camera: "},
            {"type": "video", "video": views[1]},
            {"type": "text", "text": "\nWrist camera: "},
            {"type": "video", "video": views[2]},
        ]
    elif robot_type == VLABENCH:
        content = [
            {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
            {"type": "image", "image": views[0]},
            {"type": "text", "text": "\n# Base View\n"},
            {"type": "image", "image": views[1]},
            {"type": "text", "text": "\n# Left-Wrist View\n"},
            {"type": "image", "image": views[2]},
        ]
    else:
        raise ValueError(f"unsupported XR-1 robot type {robot_type!r}")
    separator = "\n\n" if robot_type == ROBOCASA365 else "\n"
    content.append(
        {
            "type": "text",
            "text": f"{separator}Generate robot actions for the task:\n{instruction} /no_cot",
        }
    )
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


class XR1ModelServer(PredictModelServer):
    """Serve a released XR-1 checkpoint directly through the harness WebSocket."""

    def __init__(
        self,
        model_path: str,
        robot_type: str,
        *,
        revision: str | None = None,
        request_seed: int | None = None,
        chunk_size: int = 10,
        **kwargs: Any,
    ) -> None:
        if robot_type not in ROBOT_TYPES:
            raise ValueError(f"robot_type must be one of {ROBOT_TYPES}, got {robot_type!r}")
        super().__init__(chunk_size=chunk_size, **kwargs)
        self.model_path = model_path
        self.robot_type = robot_type
        self.revision = revision
        self.request_seed = request_seed
        self._histories: dict[str, deque[Observation]] = {}

        from vla_eval.dirs import require_model_available

        require_model_available(model_path)

        import torch
        from transformers import AutoModel, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("XR-1 requires a CUDA GPU")
        load_args = {"revision": revision} if revision else {}
        logger.info("Loading XR-1 from %s revision=%s", model_path, revision or "default")
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            **load_args,
        )
        available = self._processor.list_robot_types()
        if robot_type not in available:
            raise ValueError(f"checkpoint does not support {robot_type!r}; available robot types: {available}")
        self._model = (
            AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
                **load_args,
            )
            .to("cuda", dtype=torch.bfloat16)
            .eval()
        )
        self._device = self._model.device
        logger.info("XR-1 model loaded for robot_type=%s", robot_type)

    def get_observation_params(self) -> dict[str, Any]:
        if self.robot_type == ROBOCASA:
            return {"camera_names": list(ROBOCASA_CAMERAS), "camera_size": 256}
        if self.robot_type == VLABENCH:
            return {"absolute_action": True, "gripper_threshold": VLABENCH_GRIPPER_THRESHOLD}
        return {}

    def get_action_spec(self) -> dict[str, DimSpec]:
        if self.robot_type == ROBOCASA365:
            return {
                "position": POSITION_DELTA,
                "rotation": ROTATION_AA,
                "gripper": GRIPPER_CLOSE_01,
                "base_motion": BASE_MOTION,
                "control_mode": CONTROL_MODE_01,
            }
        if self.robot_type == VLABENCH:
            return {"position": POSITION_ABSOLUTE, "rotation": ROTATION_EULER, "gripper": GRIPPER_RAW}
        return {"position": POSITION_DELTA, "rotation": ROTATION_EULER, "gripper": GRIPPER_RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_start(config, ctx)
        if self.robot_type == ROBOCASA365:
            self._histories[ctx.session_id] = deque(maxlen=7)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_end(result, ctx)
        self._histories.pop(ctx.session_id, None)

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        if self.robot_type == ROBOCASA365:
            self._histories.setdefault(ctx.session_id, deque(maxlen=7)).append(obs)
        await super().on_observation(obs, ctx)

    @staticmethod
    def _images(obs: Observation, keys: Sequence[str]) -> list[Any]:
        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise KeyError("XR-1 requires an 'images' mapping")
        missing = [key for key in keys if key not in images]
        if missing:
            raise KeyError(f"XR-1 observation is missing camera views: {missing}")
        return [images[key] for key in keys]

    @staticmethod
    def _to_pil(image: Any, *, size: int, crop_ratio: float = 1.0) -> Any:
        from PIL import Image

        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
                raise ValueError(f"expected an HWC image, got {array.shape}")
            if np.issubdtype(array.dtype, np.floating):
                maximum = float(np.nanmax(array)) if array.size else 0.0
                array = np.clip(array, 0.0, 1.0 if maximum <= 1.0 + 1e-6 else 255.0)
                if maximum <= 1.0 + 1e-6:
                    array = array * 255.0
            array = array.astype(np.uint8, copy=False)
            pil = Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array).convert("RGB")
        if pil.size != (size, size):
            pil = pil.resize((size, size), Image.Resampling.BILINEAR)
        if crop_ratio < 1.0:
            width, height = pil.size
            crop_width, crop_height = max(1, int(width * crop_ratio)), max(1, int(height * crop_ratio))
            left, top = (width - crop_width) // 2, (height - crop_height) // 2
            pil = pil.crop((left, top, left + crop_width, top + crop_height)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
        return pil

    @staticmethod
    def _flat_state(obs: Observation) -> np.ndarray:
        state = obs.get("state", obs.get("states"))
        if state is None or isinstance(state, Mapping):
            raise KeyError("XR-1 requires a flat 'state' observation")
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def _model_inputs(self, obs: Observation, ctx: SessionContext) -> dict[str, Any]:
        instruction = str(obs.get("task_description", ""))
        processor_kwargs: dict[str, Any] = {}

        if self.robot_type == ROBOCASA:
            state = self._flat_state(obs)
            if state.shape != (8,):
                raise ValueError(f"expected an 8-D RoboCasa state, got {state.shape}")
            views = [self._to_pil(image, size=256, crop_ratio=0.95) for image in self._images(obs, ROBOCASA_CAMERAS)]
            states = [state]
        elif self.robot_type == ROBOCASA365:
            history = _sample_history(list(self._histories.get(ctx.session_id, ())))
            views = [
                [
                    self._to_pil(self._images(frame, ROBOCASA365_CAMERAS)[camera], size=256, crop_ratio=0.95)
                    for frame in history
                ]
                for camera in range(3)
            ]
            states = []
            for frame in history:
                state = frame.get("state")
                if not isinstance(state, Mapping):
                    raise KeyError("XR-1 RoboCasa365 requires a named 'state' mapping")
                states.append(_robocasa365_state(state))
            processor_kwargs["do_resize"] = False
        else:
            views = [self._to_pil(image, size=480) for image in self._images(obs, VLABENCH_CAMERAS)]
            states = [_vlabench_state(self._flat_state(obs))]

        inputs = self._processor.apply_chat_template(
            _build_messages(self.robot_type, views, instruction),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=_padded_state(states),
            robot_type=self.robot_type,
            **processor_kwargs,
        )
        payload = dict(inputs)
        payload["task_id"] = self.robot_type
        if self.request_seed is not None:
            payload["seed"] = self.request_seed
        return payload

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import torch

        payload = {
            key: (
                value.to(self._device, dtype=self._model.dtype)
                if torch.is_tensor(value) and value.is_floating_point()
                else value.to(self._device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in self._model_inputs(obs, ctx).items()
        }
        with torch.inference_mode():
            outputs = self._model(**payload)
        decoded = self._processor.decode_action(outputs.actions.cpu(), robot_type=self.robot_type)
        actions = decoded.detach().cpu().float().numpy()
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] < ACTION_DIMS[self.robot_type]:
            raise RuntimeError(f"unexpected XR-1 decoded action shape {actions.shape}")
        profile_actions = actions[0, :, : ACTION_DIMS[self.robot_type]]
        if self.robot_type == VLABENCH:
            profile_actions = _vlabench_targets(profile_actions, self._flat_state(obs))
        return {"actions": profile_actions}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(XR1ModelServer)
