# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
OpenAI-compatible API entrypoints for vLLM-Omni.

Provides:
- omni_run_server: Main server entry point (auto-detects model type)
- omni_run_diffusion_server: Server for diffusion models
- OmniOpenAIServingChat: Unified chat completion handler for both LLM and diffusion models
"""

from vllm_omni.entrypoints.openai.api_server import (
    build_async_diffusion,
    build_async_omni,
    omni_diffusion_init_app_state,
    omni_init_app_state,
    omni_run_diffusion_server,
    omni_run_server,
)
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

__all__ = [
    # Server functions
    "omni_run_server",
    "omni_run_diffusion_server",
    "build_async_omni",
    "build_async_diffusion",
    "omni_init_app_state",
    "omni_diffusion_init_app_state",
    # Serving classes
    "OmniOpenAIServingChat",
]
