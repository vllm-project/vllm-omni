# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
from functools import cache

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

logger = init_logger(__name__)

# environment variable value -> backend module and class
_BACKEND_CONFIG = {
    "FLASH_ATTN": {
        "module": "vllm_omni.diffusion.attention.backends.flash_attn",
        "class": "FlashAttentionBackend",
    },
    "TORCH_SDPA": {
        "module": "vllm_omni.diffusion.attention.backends.sdpa",
        "class": "SDPABackend",
    },
    "SAGE_ATTN": {
        "module": "vllm_omni.diffusion.attention.backends.sage_attn",
        "class": "SageAttentionBackend",
    },
    "ASCEND": {"module": "vllm_omni.diffusion.attention.backends.ascend_attn", "class": "AscendAttentionBackend"},
}

_BACKENDS_SUPPORT_ATTENTION_MASK = ["SDPA", "ASCEND", "FLASH_ATTN"]


def load_backend(backend_name: str) -> type[AttentionBackend]:
    config = _BACKEND_CONFIG[backend_name]

    try:
        module = importlib.import_module(config["module"])
        backend_class = getattr(module, config["class"])
        return backend_class
    except ImportError as e:
        raise ImportError(f"Failed to import module {config['module']}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class {config['class']} not found in module: {e}")


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
        if backend_name_upper not in _BACKEND_CONFIG:
            valid_backends = list(_BACKEND_CONFIG.keys())
            raise ValueError(
                f"Invalid attention backend for diffusion: '{backend_name}'. Valid backends are: {valid_backends}"
            )
        logger.info(f"Using attention backend '{backend_name_upper}' for diffusion")
        return load_backend(backend_name_upper)

    # Default to SDPA
    return SDPABackend
