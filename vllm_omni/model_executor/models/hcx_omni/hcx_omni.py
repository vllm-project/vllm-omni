# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperCLOVAX-SEED-Omni-8B multi-stage model dispatcher.

Architecture overview
---------------------
HyperCLOVAX-SEED-Omni-8B is a 3-stage omni model:

  Stage 0  – Thinker (this module, LLM engine)
    Input : text + optional image/audio
    Output: text tokens + discrete audio codes (128606–135167)
             + discrete vision codes (135168+)
    Config: engine_output_type = "latent"

  Stage 1  – Vision Decoder  (diffusion engine)
    Input : 729 discrete vision codes from stage 0
    Output: generated image (PNG / JPEG)
    Config: model_class_name = "HyperCLOVAXVisionPipeline"

  Stage 2  – Audio Decoder   (diffusion engine)
    Input : N discrete audio codes from stage 0
    Output: 24 kHz waveform (WAV / PCM)
    Config: model_class_name = "HyperCLOVAXAudioPipeline"

Stages 1 and 2 are handled by the vLLM-Omni *diffusion* engine and do
**not** go through this LLM model registry.  This dispatcher exists
only for stage 0 so that the standard ``model_arch`` routing works.
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
class HCXOmniForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsMRoPE,
    SupportsPP,
    SupportsQuant,
):
    """Top-level HyperCLOVAX-SEED-Omni-8B model for vLLM-Omni.

    This class is the ``model_arch`` entry point for the thinker stage.
    It delegates all logic to :class:`~vllm.model_executor.models.hcx_omni.
    HCXOmniForCausalLM` from the vLLM base repository.

    The vision decoder and audio decoder stages use ``model_class_name``
    (diffusion engine) and therefore do not require an entry here.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self._model = HCXOmniForCausalLM(vllm_config=vllm_config, prefix=prefix)

    # ------------------------------------------------------------------ #
    # Delegate interface implementations to the inner model               #
    # ------------------------------------------------------------------ #

    @property
    def config(self):
        return self._model.config

    # SupportsMRoPE
    def get_mrope_input_positions(self, *args: Any, **kwargs: Any):
        return self._model.get_mrope_input_positions(*args, **kwargs)

    def iter_mm_grid_thw(self, *args: Any, **kwargs: Any):
        return self._model.iter_mm_grid_thw(*args, **kwargs)

    # SupportsMultiModal
    def get_multimodal_embeddings(self, **kwargs: Any):
        return self._model.get_multimodal_embeddings(**kwargs)

    def embed_multimodal(self, **kwargs: Any):
        # Required by vLLM's multimodal embedding path; the base class
        # SupportsMultiModal declares get_multimodal_embeddings(), but some
        # call-sites (e.g. GPUModelRunner) also call embed_multimodal() directly
        # and expect a tensor rather than the raw embedding list.
        return self._model.embed_multimodal(**kwargs)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        return self._model.get_input_embeddings(input_ids, multimodal_embeddings)

    # SupportsPP
    def make_empty_intermediate_tensors(self, *args: Any, **kwargs: Any):
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

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loaded = self._model.load_weights(weights)
        # The model stores sub-modules under self._model, so vllm's named_parameters()
        # sees "_model.<name>" while self._model.load_weights() returns "<name>".
        # Re-prefix so the strict-loading check in default_loader.py matches.
        if loaded is not None:
            return {"_model." + name for name in loaded}
        return loaded

    def get_mm_mapping(self):
        return self._model.get_mm_mapping()
