"""
Autoregressive (AR) stage scaffold for MammothModa2 in vLLM-Omni.

This class will wrap the HF Mammothmoda2 LLM component (Qwen3-VL based)
to produce AR tokens / hidden states that will be consumed by the DiT stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput


class MammothModa2ARForConditionalGeneration(nn.Module):
    """Placeholder AR stage. Implements the expected interface only."""

    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.model_stage = "ar"

        # TODO: load / wrap Mammothmoda2 LLM here
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, *args, **kwargs) -> OmniOutput:
        raise NotImplementedError("AR stage not implemented yet for MammothModa2.")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("compute_logits not implemented for MammothModa2 AR scaffold.")
