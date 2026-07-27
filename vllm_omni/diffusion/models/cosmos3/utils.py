# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.models.progress_bar import _is_rank_zero

logger = init_logger(__name__)

COSMOS3_DEFAULT_CONDITION_FRAME_INDEXES_VISION = (0, 1)
COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP = "first"
# Mirrors the WAN VAE's temporal compression. Authoritative value is
# ``self.vae.config.scale_factor_temporal`` at runtime; this constant exists so
# off-line / API code that runs before the pipeline is constructed can compute
# pixel-frame budgets without instantiating the VAE.
COSMOS3_VAE_TEMPORAL_COMPRESSION = 4

ROBOLAB_DEFAULT_CONDITIONING_FPS = 15.0
ROBOLAB_DEFAULT_ACTION_CHUNK_SIZE = 32
ROBOLAB_DEFAULT_IMAGE_HEIGHT = 540
ROBOLAB_DEFAULT_IMAGE_WIDTH = 640
ROBOLAB_DEFAULT_RAW_ACTION_DIM = 8
ROBOLAB_DEFAULT_DOMAIN_NAME = "droid_lerobot"
ROBOLAB_DEFAULT_RESOLUTION = "480"
ROBOLAB_DEFAULT_GUIDANCE_SCALE = 3.0
ROBOLAB_DEFAULT_NUM_INFERENCE_STEPS = 4
ROBOLAB_DEFAULT_FLOW_SHIFT = 5.0
ROBOLAB_DEFAULT_SEED = 0
ROBOLAB_DEFAULT_ACTION_SPACE = "joint_pos"
ROBOLAB_MIDTRAIN_RAW_ACTION_DIM = 10
ROBOLAB_MIDTRAIN_POSE_ACTION_DIM = 9  # xyz position + rot6d orientation
ROBOLAB_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite "
    "sides, with the robot visible."
)


@dataclass(frozen=True)
class RoboLabPolicyInputs:
    prompt: str
    video_tensor: torch.Tensor
    action_tensor: torch.Tensor
    action_condition_indexes: list[int]
    action_start_frame_offset: int
    raw_action_dim: int
    domain_id: int
    fps: float
    height: int
    width: int
    image_size: Any
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    flow_shift: float
    seed: int
    history_length: int
    action_space: str
    observation: dict[str, Any]


@dataclass(frozen=True)
class RoboLabActionPostprocessInputs:
    history_length: int
    action_space: str
    eef_pos: np.ndarray | None = None
    eef_quat: np.ndarray | None = None


def make_robolab_action_postprocess_inputs(inputs: RoboLabPolicyInputs) -> RoboLabActionPostprocessInputs:
    if inputs.action_space != "midtrain":
        return RoboLabActionPostprocessInputs(
            history_length=inputs.history_length,
            action_space=inputs.action_space,
        )

    obs = inputs.observation
    return RoboLabActionPostprocessInputs(
        history_length=inputs.history_length,
        action_space=inputs.action_space,
        eef_pos=ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3),
        eef_quat=ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4),
    )


def normalize_condition_frame_indexes_vision(value: Any) -> tuple[int, ...]:
    """Normalize Cosmos3 vision-conditioning latent frame indexes."""
    if value is None:
        return COSMOS3_DEFAULT_CONDITION_FRAME_INDEXES_VISION
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, int):
        value = [value]

    if not isinstance(value, Iterable):
        raise TypeError(
            "Cosmos3 condition_frame_indexes_vision must be an int, comma-separated string, "
            f"or iterable of ints; got {type(value)!r}."
        )

    indexes = tuple(sorted({int(index) for index in value}))
    if not indexes:
        raise ValueError("Cosmos3 condition_frame_indexes_vision must contain at least one index.")
    if any(index < 0 for index in indexes):
        raise ValueError(f"Cosmos3 condition_frame_indexes_vision must be non-negative, got {indexes}.")
    return indexes


def condition_pixel_frame_count(
    condition_frame_indexes_vision: Iterable[int],
    temporal_compression: int = COSMOS3_VAE_TEMPORAL_COMPRESSION,
) -> int:
    return max(condition_frame_indexes_vision) * int(temporal_compression) + 1


def normalize_condition_video_keep(value: Any) -> str:
    keep = str(value or COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP).strip().lower()
    if keep not in {"first", "last"}:
        raise ValueError("Cosmos3 condition_video_keep must be either 'first' or 'last'.")
    return keep


def normalize_robolab_action_space(value: Any) -> str:
    action_space = str(value or ROBOLAB_DEFAULT_ACTION_SPACE).strip().lower()
    aliases = {
        "jointpos": "joint_pos",
        "joint_pos": "joint_pos",
        "abs_ik": "midtrain",
        "midtrain": "midtrain",
    }
    if action_space not in aliases:
        raise ValueError(f"Unsupported RoboLab action_space={value!r}; expected joint_pos/jointpos or midtrain/abs_ik.")
    return aliases[action_space]


def ensure_rgb_uint8_image(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key!r} must have shape [H, W, 3], got {image.shape}.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def ensure_2d_float_array(value: Any, key: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{key!r} must have shape [T, D] or [D], got {array.shape}.")
    if width is not None and array.shape[-1] != width:
        raise ValueError(f"{key!r} must have width {width}, got {array.shape[-1]}.")
    return np.ascontiguousarray(array)


def ensure_gripper_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[-1] != 1:
        raise ValueError(f"'observation/gripper_position' must have shape [T, 1], [T], or scalar, got {array.shape}.")
    return np.ascontiguousarray(array)


def resize_rgb_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return np.clip(np.round(resized.squeeze(0).permute(1, 2, 0).numpy()), 0, 255).astype(np.uint8)


def compose_robolab_views(obs: dict[str, Any]) -> np.ndarray | None:
    required_keys = (
        "observation/wrist_image_left",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
    )
    if not all(key in obs for key in required_keys):
        return None

    wrist = ensure_rgb_uint8_image(obs["observation/wrist_image_left"], "observation/wrist_image_left")
    left_raw = ensure_rgb_uint8_image(obs["observation/exterior_image_1_left"], "observation/exterior_image_1_left")
    right_raw = ensure_rgb_uint8_image(obs["observation/exterior_image_2_left"], "observation/exterior_image_2_left")
    half_h, half_w = wrist.shape[0] // 2, wrist.shape[1] // 2
    left = resize_rgb_uint8(left_raw, (half_h, half_w))
    right = resize_rgb_uint8(right_raw, (half_h, half_w))
    return np.concatenate([wrist, np.concatenate([left, right], axis=1)], axis=0)


def extract_robolab_image(obs: dict[str, Any]) -> np.ndarray:
    if "observation/image" in obs:
        return ensure_rgb_uint8_image(obs["observation/image"], "observation/image")
    image = compose_robolab_views(obs)
    if image is not None:
        return image
    raise ValueError("Observation must contain 'observation/image' or RoboLab wrist/exterior image keys.")


def extract_robolab_prompt_image(prompt_data: Any | None) -> np.ndarray | None:
    if not isinstance(prompt_data, dict):
        return None
    multi_modal_data = prompt_data.get("multi_modal_data", {}) or {}
    image = multi_modal_data.get("image")
    if image is None:
        return None
    if isinstance(image, PIL.Image.Image):
        return np.asarray(image.convert("RGB"))
    return ensure_rgb_uint8_image(image, "multi_modal_data.image")


def lazy_import(module_name: str, symbol_name: str, error_message: str):
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(error_message) from exc
    return getattr(module, symbol_name)


def lazy_action_transform_pipeline(max_action_dim: int):
    ActionTransformPipeline = lazy_import(
        "cosmos_framework.data.vfm.action.transforms",
        "ActionTransformPipeline",
        "Cosmos3 RoboLab policy serving requires cosmos_framework on PYTHONPATH so the "
        "golden ActionTransformPipeline can be reused.",
    )
    return ActionTransformPipeline(max_action_dim=max_action_dim, cfg_dropout_rate=0.0)


def build_robolab_unipc_scheduler(num_steps: int, shift: float, device: torch.device):
    FlowUniPCMultistepScheduler = lazy_import(
        "cosmos_framework.model.vfm.diffusion.samplers.fm_solvers_unipc",
        "FlowUniPCMultistepScheduler",
        (
            "Cosmos3 RoboLab policy serving requires cosmos_framework on PYTHONPATH so the "
            "golden FlowUniPCMultistepScheduler can be reused."
        ),
    )

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(num_steps, device=device, shift=float(shift))
    return scheduler


def convert_midtrain_rotation(value: Any, src: str, dst: str) -> np.ndarray:
    convert_rotation = lazy_import(
        "cosmos_framework.data.vfm.action.pose_utils",
        "convert_rotation",
        "Cosmos3 RoboLab midtrain action serving requires cosmos_framework pose_utils on PYTHONPATH.",
    )
    return convert_rotation(value, src, dst)


def pose_abs_to_rel(*args, **kwargs) -> np.ndarray:
    pose_abs_to_rel_func = lazy_import(
        "cosmos_framework.data.vfm.action.pose_utils",
        "pose_abs_to_rel",
        "Cosmos3 RoboLab midtrain action serving requires cosmos_framework pose_utils on PYTHONPATH.",
    )
    return pose_abs_to_rel_func(*args, **kwargs)


def pose_rel_to_abs(*args, **kwargs) -> np.ndarray:
    pose_rel_to_abs_func = lazy_import(
        "cosmos_framework.data.vfm.action.pose_utils",
        "pose_rel_to_abs",
        "Cosmos3 RoboLab midtrain action serving requires cosmos_framework pose_utils on PYTHONPATH.",
    )
    return pose_rel_to_abs_func(*args, **kwargs)


def build_abs_pose_from_components(*args, **kwargs) -> np.ndarray:
    build_abs_pose_from_components_func = lazy_import(
        "cosmos_framework.data.vfm.action.pose_utils",
        "build_abs_pose_from_components",
        "Cosmos3 RoboLab midtrain action serving requires cosmos_framework pose_utils on PYTHONPATH.",
    )
    return build_abs_pose_from_components_func(*args, **kwargs)


def next_robolab_seed(extra: dict[str, Any], obs: dict[str, Any], request_id: str | None) -> int:
    base_seed = int(extra.get("robolab_seed") or ROBOLAB_DEFAULT_SEED)
    deterministic_seed = str(extra.get("deterministic_seed", "")).strip().lower() in {"1", "true", "yes", "on"}
    if deterministic_seed:
        return base_seed
    explicit_seed = extra.get("seed")
    if explicit_seed is not None:
        return int(explicit_seed)
    seed_key = "|".join(
        str(part)
        for part in (
            base_seed,
            extra.get("session_id", ""),
            request_id or "",
            obs.get("prompt", ""),
        )
    )
    return zlib.crc32(seed_key.encode("utf-8")) & 0x7FFFFFFF


def log_robolab_action_summary(label: str, value: Any) -> None:
    if not _is_rank_zero():
        return
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(array)
    if finite.any():
        finite_min = float(array[finite].min())
        finite_max = float(array[finite].max())
    else:
        finite_min = None
        finite_max = None
    if array.ndim == 0:
        head = array.reshape(1).tolist()
    else:
        head = array.reshape(-1, array.shape[-1])[:3].tolist()
    logger.info(
        "RoboLab action summary %s: shape=%s nan=%d finite=%d finite_min=%s finite_max=%s head=%s",
        label,
        tuple(array.shape),
        int(np.isnan(array).sum()),
        int(finite.sum()),
        finite_min,
        finite_max,
        head,
    )


def postprocess_robolab_action(action: torch.Tensor, inputs: RoboLabActionPostprocessInputs) -> np.ndarray:
    action_np = action[0].float().cpu().numpy()
    log_robolab_action_summary("raw_model_action", action_np)
    history_length = int(inputs.history_length)
    action_np = action_np[history_length:]
    action_np[:, -1] = 1.0 - action_np[:, -1]

    if inputs.action_space == "midtrain":
        if inputs.eef_pos is None or inputs.eef_quat is None:
            raise ValueError("RoboLab midtrain action postprocess requires eef_pos and eef_quat metadata.")
        initial_pose = np.eye(4, dtype=np.float32)
        initial_pose[:3, :3] = convert_midtrain_rotation(inputs.eef_quat[-1], "quat_xyzw", "matrix")
        initial_pose[:3, 3] = inputs.eef_pos[-1]
        abs_pose = pose_rel_to_abs(
            action_np[:, :ROBOLAB_MIDTRAIN_POSE_ACTION_DIM],
            rotation_format="rot6d",
            pose_convention="backward_framewise",
            initial_pose=initial_pose,
        )
        position = abs_pose[1:, :3, 3]
        quat_xyzw = convert_midtrain_rotation(abs_pose[1:, :3, :3], "matrix", "quat_xyzw")
        action_np = np.concatenate([position, quat_xyzw, action_np[:, ROBOLAB_MIDTRAIN_POSE_ACTION_DIM:]], axis=-1)

    log_robolab_action_summary("postprocessed_robolab_action", action_np)
    return np.asarray(action_np, dtype=np.float32)
