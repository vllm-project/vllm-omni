# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preprocessing for the π0 VLA model.

Converts a raw robot observation (multi-camera images + language instruction +
proprioceptive state) into the tensors that ``Pi0ForActionPrediction.sample_actions``
consumes:

  * images       : list of ``(1, 3, 224, 224)`` float tensors in ``[-1, 1]``,
                   one per camera slot (real cameras first, then ``-1``-filled
                   padding) in the checkpoint's ``image_feature_keys`` order
  * image_masks  : list of ``(1,)`` bool tensors, ``True`` = real camera
  * lang_tokens  : ``(1, L)`` long, PaliGemma tokenizer, ``\\n``-terminated
  * lang_masks   : ``(1, L)`` bool
  * state        : ``(1, max_state_dim)`` float32, zero-padded

Image/prompt preprocessing matches OpenPI and LeRobot bit-for-bit — see the
parity test (``test_pi0_parity.py``). A set of stateless helpers the diffusion
pipeline calls directly (the DreamZero contract: the pipeline owns its
preprocessing).

Reference:
  - OpenPI: openpi/src/openpi/shared/image_tools.py (resize_with_pad)
  - OpenPI: openpi/src/openpi/models_pytorch/preprocessing_pytorch.py
  - LeRobot: lerobot/src/lerobot/policies/pi0/modeling_pi0.py (Pi0NewLineProcessor)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Defaults straight from the π0 reference configs.
PI0_IMAGE_SIZE = 224  # openpi/models/model.py IMAGE_RESOLUTION
PI0_NUM_IMAGE_TOKENS = 256  # SigLIP So400m/14 on 224×224 → (224/14)**2
PI0_IMAGE_TOKEN_INDEX = 257152  # openpi/models_pytorch/gemma_pytorch.py
PI0_MAX_CAMERAS = 3  # openpi/models/model.py (3 camera slots)
PI0_MAX_TOKEN_LEN = 48  # openpi/models/pi0_config.py


def resize_with_pad(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize ``(B, C, H, W)`` images to the target shape, preserving aspect
    ratio with -1 padding on the short side.

    Matches openpi ``image_tools.resize_with_pad_torch`` — the clamp to
    [-1, 1] is what lets the padded region blend with SigLIP-normalized
    pixels without adding signal at the boundary.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4-D (B,C,H,W), got {images.ndim}-D")
    _, _, cur_h, cur_w = images.shape
    ratio = max(cur_w / target_width, cur_h / target_height)
    rh, rw = int(cur_h / ratio), int(cur_w / ratio)
    align_corners = False if mode == "bilinear" else None
    resized = F.interpolate(images, size=(rh, rw), mode=mode, align_corners=align_corners)
    resized = resized.clamp(-1.0, 1.0)
    ph, rem_h = divmod(target_height - rh, 2)
    pw, rem_w = divmod(target_width - rw, 2)
    return F.pad(resized, (pw, pw + rem_w, ph, ph + rem_h), value=-1.0)


def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL → ``(1, C, H, W)`` float32 in ``[-1, 1]`` (SigLIP normalization)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class Pi0ImageProcessor:
    """Minimal image preprocessor: image → normalized + padded ``[-1,1]`` tensor."""

    def __init__(self, image_size: int = PI0_IMAGE_SIZE):
        self.image_size = image_size

    def preprocess_single(self, image: Any) -> torch.Tensor:
        """Accept a PIL image, an HWC uint8/float ndarray, or a CHW tensor and
        return a ``(1, 3, image_size, image_size)`` float tensor in ``[-1, 1]``."""
        t = self._to_tensor(image)
        if t.shape[2] != self.image_size or t.shape[3] != self.image_size:
            t = resize_with_pad(t, self.image_size, self.image_size)
        return t

    def _to_tensor(self, image: Any) -> torch.Tensor:
        if isinstance(image, Image.Image):
            return pil_image_to_tensor(image)
        if isinstance(image, np.ndarray):
            arr = image
            # HWC → normalize uint8/[0,255] to [-1,1]; assume already [-1,1] if float.
            if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                if np.issubdtype(arr.dtype, np.integer) or arr.max() > 1.0:
                    arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
                else:
                    arr = arr.astype(np.float32)
                return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return torch.as_tensor(arr, dtype=torch.float32)
        if isinstance(image, torch.Tensor):
            t = image
            if t.ndim == 3:  # (C, H, W) → (1, C, H, W)
                t = t.unsqueeze(0)
            return t.to(dtype=torch.float32)
        raise TypeError(f"Unsupported image type for π0 preprocessing: {type(image)}")

    def make_empty_image(self) -> torch.Tensor:
        """Fill tensor for an unused camera slot — pure -1, matches OpenPI/LeRobot."""
        return torch.full((1, 3, self.image_size, self.image_size), -1.0)


def tokenize_prompt(tokenizer, text: str, max_token_len: int = PI0_MAX_TOKEN_LEN):
    """Return ``(input_ids, attention_mask)`` lists, length ``max_token_len``.

    Matches LeRobot's ``Pi0NewLineProcessor``: append ``\\n`` if the caller
    didn't, then run the PaliGemma tokenizer with right-padding.
    """
    prompt = text or ""
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"
    enc = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_token_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors=None,
    )
    return list(enc["input_ids"]), list(enc["attention_mask"])


def _extract_images(robot_obs: dict, config) -> dict[str, Any]:
    """Pull a ``{feature_key: image}`` map out of a raw robot obs.

    Accepts two shapes:
      * ``robot_obs["images"]`` is a dict keyed by camera name; or
      * flat ``observation/<cam>`` style keys on ``robot_obs`` itself.
    Keys are translated through ``config.image_key_map`` (raw obs key →
    checkpoint feature key) so serving wire names map onto ``input_features``.
    """
    images = robot_obs.get("images")
    if not isinstance(images, dict):
        # Fall back to flat keys (anything that isn't a known scalar field).
        images = {k: v for k, v in robot_obs.items() if k not in ("state", "prompt", "session_id", "reset", "images")}
    key_map = getattr(config, "image_key_map", None) or {}
    return {key_map.get(k, k): v for k, v in images.items()}


def build_model_inputs(robot_obs: dict, config, tokenizer, device: torch.device):
    """Convert a raw robot observation into ``sample_actions`` inputs.

    Reproduces the LeRobot camera ordering exactly: iterate
    ``config.image_feature_keys`` (the ordered camera identities from the
    checkpoint's ``input_features``); for each, use the supplied image (mask
    ``True``) or a ``-1``-filled empty image (mask ``False``). Then tokenize
    the prompt and zero-pad/truncate the state to ``max_state_dim``.

    Returns ``(images, image_masks, lang_tokens, lang_masks, state)`` with a
    leading batch dim of 1 on the tensors.
    """
    image_size = int(config.image_resolution[0])
    img_proc = Pi0ImageProcessor(image_size=image_size)

    feature_keys = config.image_feature_keys or []
    if not feature_keys:
        # No declared camera order — fall back to whatever the obs provides,
        # capped at max_cameras (preserves insertion order).
        provided = _extract_images(robot_obs, config)
        feature_keys = list(provided.keys())[: config.max_cameras]

    obs_images = _extract_images(robot_obs, config)

    images: list[torch.Tensor] = []
    image_masks: list[torch.Tensor] = []
    for key in feature_keys[: config.max_cameras]:
        img = obs_images.get(key)
        if img is not None:
            tensor = img_proc.preprocess_single(img).to(device=device)
            mask = True
        else:
            tensor = img_proc.make_empty_image().to(device=device)
            mask = False
        images.append(tensor)
        image_masks.append(torch.tensor([mask], dtype=torch.bool, device=device))

    # Pad up to max_cameras with empty slots if the checkpoint declares fewer
    # cameras than the model attends to.
    while len(images) < config.max_cameras:
        images.append(img_proc.make_empty_image().to(device=device))
        image_masks.append(torch.tensor([False], dtype=torch.bool, device=device))

    prompt = robot_obs.get("prompt", "") or ""
    ids, attn = tokenize_prompt(tokenizer, prompt, config.tokenizer_max_length)
    lang_tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lang_masks = torch.tensor([attn], dtype=torch.bool, device=device)

    state = _build_state(robot_obs.get("state"), config.max_state_dim, device)

    return images, image_masks, lang_tokens, lang_masks, state


def _build_state(raw_state, max_state_dim: int, device: torch.device) -> torch.Tensor:
    """Zero-pad / truncate the raw state to ``(1, max_state_dim)`` float32."""
    if raw_state is None:
        return torch.zeros(1, max_state_dim, dtype=torch.float32, device=device)
    state = torch.as_tensor(raw_state, dtype=torch.float32, device=device)
    if state.dim() == 1:
        state = state[None, :]
    if state.shape[-1] < max_state_dim:
        state = F.pad(state, (0, max_state_dim - state.shape[-1]))
    elif state.shape[-1] > max_state_dim:
        state = state[..., :max_state_dim]
    return state
