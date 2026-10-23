# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-boundary checks shared by the LingBot World ComfyUI nodes."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

import torch

from .utils.format import image_tensor_to_base64

#: The causal-fast checkpoint the stepwise deploy config is built for.
DEFAULT_MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
#: The streaming video endpoint the nodes connect to by default.
DEFAULT_ENDPOINT = "ws://127.0.0.1:8000/v1/realtime/video"
#: Server-side limit on a ``session.start`` message.
MAX_START_MESSAGE_BYTES = 4 * 1024 * 1024


def validate_endpoint(server_url: str) -> str:
    """Accept a ws:// or wss:// endpoint without credentials or a fragment."""
    url = server_url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("ws", "wss") or not parts.hostname or parts.username or parts.password or parts.fragment:
        raise ValueError("Use a ws:// or wss:// video endpoint without embedded credentials or a fragment.")
    return url


def validate_size(width: int, height: int) -> None:
    """The deploy config fixes the cache geometry, so both sides must agree."""
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("Width and height must be positive multiples of 16 matching the Omni deploy config.")


def encode_first_frame(image: torch.Tensor, width: int, height: int) -> str:
    """Turn one ComfyUI IMAGE into the base64 data URL the server takes."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
        raise ValueError("Connect exactly one RGB image; select one image from a batch first.")
    image = image.detach().cpu()
    if not torch.isfinite(image).all():
        raise ValueError("The input image contains non-finite values.")
    return image_tensor_to_base64(image.clamp(0, 1), size=(width, height))


def check_start_payload(payload: dict[str, Any]) -> None:
    """Bound the start message the way the server does, in bytes."""
    size = len(json.dumps(payload).encode("utf-8"))
    if size > MAX_START_MESSAGE_BYTES:
        raise ValueError(
            f"The start message is {size} bytes, over Omni's {MAX_START_MESSAGE_BYTES}-byte limit; "
            "reduce the image dimensions or prompt."
        )
