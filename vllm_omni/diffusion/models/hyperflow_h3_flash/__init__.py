# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HyperFlow H3 Flash helpers attached to MiniMaxH3Pipeline (RFC #10179)."""

from vllm_omni.diffusion.models.hyperflow_h3_flash.checkpoint import (
    HYPERFLOW_BASE_MODEL,
    HYPERFLOW_H3_FLASH_MODEL_ID,
    looks_like_hyperflow_h3_flash,
)
from vllm_omni.diffusion.models.hyperflow_h3_flash.prepare import OFFICIAL_T2VA_PROMPT
from vllm_omni.diffusion.models.hyperflow_h3_flash.schedule import DEFAULT_STEPS, SUPPORTED_STEPS, sigma_grid

__all__ = [
    "DEFAULT_STEPS",
    "HYPERFLOW_BASE_MODEL",
    "HYPERFLOW_H3_FLASH_MODEL_ID",
    "OFFICIAL_T2VA_PROMPT",
    "SUPPORTED_STEPS",
    "looks_like_hyperflow_h3_flash",
    "sigma_grid",
]
