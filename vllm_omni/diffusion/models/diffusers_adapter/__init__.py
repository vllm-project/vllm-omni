# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Diffusers backend adapter for vLLM-Omni."""

from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
    DiffusersAdapterPipeline,
)

__all__ = [
    "DiffusersAdapterPipeline",
]
