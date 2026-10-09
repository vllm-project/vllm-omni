# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox as one registered architecture with two stages.

The pipeline names this class for every stage; ``model_stage`` picks the
stage class it wraps. What vLLM inspects on the registered class itself
(``embed_input_ids``, ``compute_logits``, ``forward``, ``load_weights``,
``make_empty_intermediate_tensors``) is defined here and delegates. Every
other attribute the runner probes for (``has_preprocess``, ``preprocess``,
``have_multimodal_outputs``, ``enable_update_additional_information``,
``requires_request_ids``, ``allow_patterns_overrides``,
``on_requests_finished``) resolves on the stage through ``__getattr__``, so
a flag set on a stage cannot be forgotten here.
"""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import ChatterboxS3Gen
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import ChatterboxT3ForConditionalGeneration
from vllm_omni.model_executor.models.output_templates import OmniOutput

# Deploy ``model_stage`` to the class that runs it.
STAGES: dict[str, type[nn.Module]] = {
    "chatterbox_t3": ChatterboxT3ForConditionalGeneration,
    "chatterbox_s3gen": ChatterboxS3Gen,
}


class ChatterboxForConditionalGeneration(nn.Module, SupportsPP):
    """Stage 0 (``chatterbox_t3``) or stage 1 (``chatterbox_s3gen``)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        stage = vllm_config.model_config.model_stage
        if stage not in STAGES:
            raise ValueError(f"unknown Chatterbox model_stage {stage!r}; expected one of {sorted(STAGES)}")
        self.model = STAGES[stage](vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

    def __getattr__(self, name: str) -> object:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Stage 0's speech-token embedding.

        Defined on this class because vLLM decides a model generates text,
        and so gives it the generate runner, by finding ``embed_input_ids``,
        ``forward`` and ``compute_logits`` on the registered class.
        """
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Stage 0's speech logits."""
        return self.model.compute_logits(hidden_states)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        """Stage 0's pipeline-parallel buffers."""
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **runner_kwargs: object,
    ) -> OmniOutput | IntermediateTensors:
        """Run the stage."""
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **runner_kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the stage's checkpoint file.

        Returns:
            The names of every parameter filled, under this module's prefix.
        """
        return {f"model.{name}" for name in self.model.load_weights(weights)}
