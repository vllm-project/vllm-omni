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
from .latent_forms import LATENT_FORMS, LatentForm, lookup_latent_form
from .pid_model import PidInferenceModel
from .runner_integration import (
    PidPassthrough,
    decode_stepwise_output,
    decode_with_pid,
    init_pid_decoder_on,
    maybe_pid_passthrough,
    stepwise_pid_active,
)

__all__ = [
    "PidInferenceModel",
    "load_pid_checkpoint",
    "PidNetConfig",
    "PidSamplingConfig",
    "PidDecodeConfig",
    "PidDecoder",
    "LatentForm",
    "LATENT_FORMS",
    "lookup_latent_form",
    "init_pid_decoder_on",
    "decode_with_pid",
    "maybe_pid_passthrough",
    "PidPassthrough",
    "stepwise_pid_active",
    "decode_stepwise_output",
    "QWENIMAGE_PID_NET_CONFIG",
    "FLUX_PID_NET_CONFIG",
    "SD3_PID_NET_CONFIG",
    "SDXL_PID_NET_CONFIG",
    "FLUX2_PID_NET_CONFIG",
    "PID_SAMPLING_CONFIG",
    "get_pid_net_config",
    "get_pid_sampling_config",
]
