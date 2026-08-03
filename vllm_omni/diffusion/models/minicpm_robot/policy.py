# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-RobotManip policy for VLA inference via AutoModel.from_pretrained."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

DEFAULT_PROMPT_TEMPLATE = (
    "The robot is LIBERO Franka, a simulated single-arm Franka manipulator. "
    "Its action control method is absolute single-arm end-effector pose in the unified 80D layout "
    "with gripper closed command, and its action FPS is 20 Hz. Task: {instruction}"
)
STATE_DIM = 80


def _format_prompt(template: str, instruction: str) -> str:
    """Apply prompt template without choking on braces in the instruction."""
    if "{instruction}" in template:
        return template.replace("{instruction}", instruction)
    return f"{template}{instruction}"


def _load_images(images: Any) -> list[np.ndarray]:
    """Normalize camera inputs to HxWx3 uint8 arrays.

    Accepts a list/tuple of images, or a dict of camera_name -> image
    (OpenPI-style). Spatial resizing is left to the MiniCPM-V processor,
    which infers a target size from each image's original aspect ratio.
    """
    if images is None:
        raise ValueError("robot_obs['images'] is required.")
    if isinstance(images, Mapping):
        image_list = list(images.values())
    elif isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
        image_list = list(images)
    else:
        raise ValueError(f"robot_obs['images'] must be a list or dict of images, got {type(images).__name__}.")
    if not image_list:
        raise ValueError("robot_obs['images'] must contain at least one image.")

    loaded: list[np.ndarray] = []
    for image in image_list:
        if isinstance(image, (str, Path)):
            with Image.open(image) as pil_image:
                array = np.asarray(pil_image.convert("RGB"))
        elif isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            array = image
        elif isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            # CHW -> HWC
            array = np.transpose(array, (1, 2, 0))
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected an HxWx3 image, got shape {array.shape}")
        if array.dtype != np.uint8:
            max_val = float(np.max(array)) if array.size else 0.0
            if max_val <= 1.0:
                array = (array * 255.0).clip(0, 255).astype(np.uint8)
            else:
                array = np.asarray(array).clip(0, 255).astype(np.uint8)

        loaded.append(np.ascontiguousarray(array))
    return loaded


def _prepare_state(state: Any, *, state_dim: int = STATE_DIM) -> np.ndarray:
    if state is None:
        raise ValueError("robot_obs['state'] is required.")
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_arr.shape[0] != state_dim:
        raise ValueError(f"robot_obs['state'] must have shape ({state_dim},), got {tuple(np.asarray(state).shape)}.")
    return state_arr


def normalize_robot_obs(
    observation: Mapping[str, Any],
    *,
    state_dim: int = STATE_DIM,
) -> dict[str, Any]:
    """Validate and normalize OpenPI / adapter observations."""
    if "images" not in observation and "video" not in observation:
        raise ValueError("robot_obs must include 'images' (or 'video').")
    images = observation.get("images", observation.get("video"))
    language = observation.get("language")
    if language is None:
        language = observation.get("prompt", "")
    normalized: dict[str, Any] = {
        "images": _load_images(images),
        "state": _prepare_state(observation.get("state"), state_dim=state_dim),
        "language": str(language),
    }
    if "embodiment_id" in observation and observation["embodiment_id"] is not None:
        normalized["embodiment_id"] = int(observation["embodiment_id"])
    return normalized


class MiniCPMRobotPolicy:
    """MiniCPM-RobotManip policy backed by a complete HF VLA model.

    The full ``MiniCPMV_VLA`` model (MiniCPM-V 4.6 VLM + DiT ActionHead) is
    loaded via ``AutoModel.from_pretrained`` with ``trust_remote_code=True``.
    Weights live in ``model.safetensors`` (3.5 GB); the pipeline does not
    participate in vLLM's weight-loading path (``weights_sources=()``).

    The prompt template can be overridden via ``model_config.prompt_template``
    in the deploy YAML.  The default targets the LIBERO Franka embodiment.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: int | str,
        processor_path: str | None = None,
        prompt_template: str | None = None,
        embodiment_id: int = 0,
    ):
        self.device = torch.device(device)
        self.embodiment_id = embodiment_id
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.state_dim = STATE_DIM

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(device=self.device)
        self.state_dim = int(getattr(self.model.config, "state_dim", STATE_DIM))

        proc_path = processor_path or model_path
        self.processor = AutoProcessor.from_pretrained(
            proc_path,
            trust_remote_code=True,
        )

    def _build_text(self, raw_instruction: str) -> str:
        if "unified 80D layout" in raw_instruction and "Task:" in raw_instruction:
            return raw_instruction
        return _format_prompt(self.prompt_template, raw_instruction)

    def _move_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, Mapping):
            return {key: self._move_to_device(item) for key, item in value.items()}
        return value

    @torch.no_grad()
    def get_action(
        self,
        observation: Mapping[str, Any],
        *,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Run a single VLA inference step.

        Args:
            observation:
                ``{"images": <list|dict>, "state": <(state_dim,)>,
                  "language"|"prompt": <str>}``
            seed: Optional RNG seed for the action-head diffusion sampler.

        Returns:
            ``{"default": <np.ndarray (1, action_horizon, action_dim)>}``
        """
        obs = normalize_robot_obs(observation, state_dim=self.state_dim)
        images = obs["images"]
        robot_state = obs["state"]
        text = self._build_text(obs["language"])

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": text},
                ],
            }
        ]
        chat_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": False},
        )
        # Match the official HF example: move every tensor/list-of-tensors.
        vlm_inputs = {key: self._move_to_device(value) for key, value in chat_inputs.items()}
        if "input_ids" in vlm_inputs and isinstance(vlm_inputs["input_ids"], torch.Tensor):
            vlm_inputs["input_ids"] = vlm_inputs["input_ids"].long()
        if "attention_mask" in vlm_inputs and "language_attention_mask" not in vlm_inputs:
            mask = vlm_inputs.pop("attention_mask")
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            vlm_inputs["language_attention_mask"] = mask

        state_t = torch.as_tensor(robot_state, device=self.device).float()
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        state_t = state_t.unsqueeze(1)  # (B, state_dim) -> (B, 1, state_dim)
        emb_id = obs.get("embodiment_id", self.embodiment_id)
        emb_id_t = torch.tensor([int(emb_id)], dtype=torch.long, device=self.device)

        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        actions = self.model.predict_action(
            state=state_t,
            embodiment_id=emb_id_t,
            **vlm_inputs,
        )
        return {
            "default": actions.cpu().numpy().astype(np.float32),
        }

    def reset(self) -> dict[str, Any]:
        """Reset policy state (markovian, no persistent state)."""
        return {}
