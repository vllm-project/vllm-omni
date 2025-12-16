"""
Diffusion (DiT + VAE) stage scaffold for MammothModa2 in vLLM-Omni.

This class will host the diffusion transformer and VAE decoding logic that
consumes AR outputs and produces images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput


class MammothModa2DiTForConditionalGeneration(nn.Module):
    """Placeholder DiT stage. Implements the expected interface only."""

    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.model_stage = "dit"

        # TODO: load diffusion transformer + VAE here
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, *args, **kwargs) -> OmniOutput:
        raise NotImplementedError("DiT stage not implemented yet for MammothModa2.")
