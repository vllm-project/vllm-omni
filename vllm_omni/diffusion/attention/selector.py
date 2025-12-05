# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from functools import cache

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

logger = init_logger(__name__)

# environment variable value -> backend class
SUPPORTED_BACKENDS = {
    "FLASH_ATTN": FlashAttentionBackend,
    "TORCH_SDPA": SDPABackend,
}


@cache
def get_attn_backend(head_size: int) -> type[AttentionBackend]:
    """
    Get attention backend for diffusion models.

    The backend is selected based on the following priority:
    1. DIFFUSION_ATTENTION_BACKEND environment variable (if set, e.g. export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN)
    2. Default backend (SDPA)

    Args:
        head_size: Head size (currently not used for selection, but kept for API compatibility)

    Returns:
        The selected attention backend class
    """
    # Check environment variable
    backend_name: str | None = os.environ.get("DIFFUSION_ATTENTION_BACKEND")

    if backend_name is not None:
        backend_name_upper = backend_name.upper()
        if backend_name_upper not in SUPPORTED_BACKENDS:
            valid_backends = list(SUPPORTED_BACKENDS.keys())
            raise ValueError(
                f"Invalid attention backend for diffusion: '{backend_name}'. Valid backends are: {valid_backends}"
            )
        logger.info(f"Using attention backend '{backend_name_upper}' for diffusion")
        return SUPPORTED_BACKENDS[backend_name_upper]

    # Default to SDPA
    return SDPABackend
