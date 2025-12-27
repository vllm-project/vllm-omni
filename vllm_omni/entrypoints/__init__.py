# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
vLLM-Omni entrypoints module.

Provides high-level interfaces for running omni models including:
- AsyncOmni: Async orchestrator for multi-stage LLM pipelines
- AsyncOmniDiffusion: Async interface for diffusion model inference
- Omni: Unified entrypoint that auto-selects between LLM and Diffusion
"""

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.omni import Omni

__all__ = [
    "AsyncOmni",
    "AsyncOmniDiffusion",
    "Omni",
]
