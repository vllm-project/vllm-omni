# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrapper around the vLLM base HCXOmniForCausalLM thinker.

Registers the multimodal processor for the vLLM-Omni pipeline context
and exposes all interfaces required by the thinker stage.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.hcx_omni import (
    HCXOmniDummyInputsBuilder,
    HCXOmniForCausalLM,
    HCXOmniMultiModalProcessor,
    HCXOmniProcessingInfo,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    HCXOmniMultiModalProcessor,
    info=HCXOmniProcessingInfo,
    dummy_inputs=HCXOmniDummyInputsBuilder,
)
class HCXOmniThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsMRoPE,
    SupportsPP,
    SupportsQuant,
):
    """Thinker stage model for HyperCLOVAX-SEED-Omni-8B.

    This is a thin wrapper around :class:`HCXOmniForCausalLM` (defined in
    the vLLM base repository) that registers the multimodal processor and
    exposes the standard vLLM model interfaces needed by the omni pipeline
    thinker stage.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self._model = HCXOmniForCausalLM(vllm_config=vllm_config, prefix=prefix)

    # --- delegate all interface methods to inner model ------------------- #

    @property
    def config(self):
        return self._model.config

    @property
    def language_model(self):
        return self._model.language_model

    @property
    def visual(self):
        return self._model.visual

    @property
    def audio_tower(self):
        return self._model.audio_tower

    # SupportsMRoPE
    def get_mrope_input_positions(self, *args, **kwargs):
        return self._model.get_mrope_input_positions(*args, **kwargs)

    def iter_mm_grid_thw(self, *args, **kwargs):
        return self._model.iter_mm_grid_thw(*args, **kwargs)

    # SupportsMultiModal
    def get_multimodal_embeddings(self, **kwargs: Any) -> MultiModalEmbeddings | None:
        return self._model.get_multimodal_embeddings(**kwargs)

    def embed_multimodal(self, **kwargs: Any) -> MultiModalEmbeddings:
        # Required by vLLM's multimodal embedding path; the base class
        # SupportsMultiModal declares get_multimodal_embeddings(), but some
        # call-sites (e.g. GPUModelRunner) also call embed_multimodal() directly
        # and expect a tensor rather than the raw embedding list.
        return self._model.embed_multimodal(**kwargs)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
    ) -> torch.Tensor:
        return self._model.get_input_embeddings(input_ids, multimodal_embeddings)

    # SupportsPP
    def make_empty_intermediate_tensors(self, *args, **kwargs):
        return self._model.make_empty_intermediate_tensors(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self._model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self._model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = self._model.load_weights(weights)
        # vLLM's default_loader validates that every named_parameter() is
        # present in the set returned by load_weights().  Because weights are
        # stored under self._model, named_parameters() yields "_model.<name>"
        # while the inner load_weights() returns bare "<name>".  Re-prefix the
        # returned set so the strict-loading check passes.
        if loaded is not None:
            return {"_model." + name for name in loaded}
        return loaded

    def get_mm_mapping(self):
        return self._model.get_mm_mapping()

    # SupportsQuant
    def get_quant_config(self):
        return getattr(self._model, "get_quant_config", lambda: None)()
