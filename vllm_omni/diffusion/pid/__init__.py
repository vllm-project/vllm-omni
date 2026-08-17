# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PiD (Pixel Diffusion) decoder for super-resolution.

Model-agnostic core -- works with any upstream LDM that produces a latent
compatible with PiD's LQ projection.
"""

from .checkpoint import load_pid_checkpoint
from .config import (
    FLUX2_PID_NET_CONFIG,
    FLUX_PID_NET_CONFIG,
    PID_SAMPLING_CONFIG,
    QWENIMAGE_PID_NET_CONFIG,
    SD3_PID_NET_CONFIG,
    SDXL_PID_NET_CONFIG,
    PidNetConfig,
    PidSamplingConfig,
    get_pid_net_config,
    get_pid_sampling_config,
)
from .decoder import PidDecodeConfig, PidDecoder
from .mixin import PidDecodeMixin
from .pid_model import PidInferenceModel

__all__ = [
    "PidInferenceModel",
    "load_pid_checkpoint",
    "PidNetConfig",
    "PidSamplingConfig",
    "PidDecodeConfig",
    "PidDecoder",
    "PidDecodeMixin",
    "QWENIMAGE_PID_NET_CONFIG",
    "FLUX_PID_NET_CONFIG",
    "SD3_PID_NET_CONFIG",
    "SDXL_PID_NET_CONFIG",
    "FLUX2_PID_NET_CONFIG",
    "PID_SAMPLING_CONFIG",
    "get_pid_net_config",
    "get_pid_sampling_config",
]
