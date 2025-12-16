"""
Minimal scaffolding for MammothModa2 support in vLLM-Omni.

This is a placeholder to wire model registry and stage configs; the actual
AR/DiT execution path will be filled in subsequent iterations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput


class MammothModa2ForConditionalGeneration(nn.Module):
    """
    Thin wrapper around MammothModa2 to align with vLLM-Omni registry.

    Current implementation only defines the interface expected by runners.
    The forward will be implemented with AR (token generation) and DiT
    decoding in later steps.
    """

    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_stage = vllm_config.model_config.model_stage
        self.prefix = prefix

        # Placeholder buffers to satisfy runner shape checks.
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, *args, **kwargs) -> OmniOutput:
        """Placeholder forward; to be replaced with real AR/DiT logic."""
        raise NotImplementedError(
            "MammothModa2ForConditionalGeneration is scaffold-only. "
            "Implement AR generate and DiT decode before enabling this model."
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("compute_logits not implemented for MammothModa2 scaffold.")
