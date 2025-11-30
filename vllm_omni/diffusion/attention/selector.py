# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

logger = init_logger(__name__)


def get_attn_backend() -> type[AttentionBackend]:
    """Get attention backend."""
    # for now, we only support SDPA backend
    # return FlashAttentionBackend
    return SDPABackend
