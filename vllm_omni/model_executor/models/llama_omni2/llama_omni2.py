# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-selecting wrapper for LLaMA-Omni 2."""

from collections.abc import Callable, Iterable
from functools import cached_property
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.sample.sampler import Sampler

from .llama_omni2_thinker import (
    LlamaOmni2DummyInputsBuilder,
    LlamaOmni2MultiModalProcessor,
    LlamaOmni2ProcessingInfo,
)


def _build_thinker(*, vllm_config: VllmConfig, prefix: str) -> nn.Module:
    from .llama_omni2_thinker import (
        LlamaOmni2ThinkerForConditionalGeneration,
    )

    return LlamaOmni2ThinkerForConditionalGeneration(
        vllm_config=vllm_config,
        prefix=prefix,
    )


def _build_talker(*, vllm_config: VllmConfig, prefix: str) -> nn.Module:
    from .llama_omni2_talker import (
        LlamaOmni2TalkerForConditionalGeneration,
    )

    return LlamaOmni2TalkerForConditionalGeneration(
        vllm_config=vllm_config,
        prefix=prefix,
    )


def _build_code2wav(*, vllm_config: VllmConfig, prefix: str) -> nn.Module:
    from .llama_omni2_code2wav import LlamaOmni2Code2Wav

    return LlamaOmni2Code2Wav(vllm_config=vllm_config, prefix=prefix)


_STAGE_FACTORIES: dict[
    str,
    Callable[..., nn.Module],
] = {
    "thinker": _build_thinker,
    "talker": _build_talker,
    "code2wav": _build_code2wav,
}


@MULTIMODAL_REGISTRY.register_processor(
    LlamaOmni2MultiModalProcessor,
    info=LlamaOmni2ProcessingInfo,
    dummy_inputs=LlamaOmni2DummyInputsBuilder,
)
class Omni2Speech2SQwen2ForCausalLM(nn.Module, SupportsMultiModal):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        from .llama_omni2_thinker import (
            LlamaOmni2ThinkerForConditionalGeneration,
        )

        return LlamaOmni2ThinkerForConditionalGeneration.get_placeholder_str(
            modality,
            i,
        )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_stage = vllm_config.model_config.model_stage
        factory = _STAGE_FACTORIES.get(self.model_stage)
        if factory is None:
            supported = ", ".join(_STAGE_FACTORIES)
            raise ValueError(f"Unsupported LLaMA-Omni 2 model stage {self.model_stage!r}; expected one of: {supported}")
        self.model = factory(vllm_config=vllm_config, prefix=prefix)
        self.make_empty_intermediate_tensors = getattr(
            self.model,
            "make_empty_intermediate_tensors",
            lambda: None,
        )
        for attribute in (
            "have_multimodal_outputs",
            "prefer_model_sampler",
            "has_preprocess",
            "has_postprocess",
            "enable_update_additional_information",
            "requires_raw_input_tokens",
            "preprocess_once_buffer_keys",
            "cumulative_postprocess_output_buffer_keys",
        ):
            if hasattr(self.model, attribute):
                setattr(self, attribute, getattr(self.model, attribute))

    @cached_property
    def sampler(self) -> Any:
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def get_language_model(self) -> nn.Module:
        getter = getattr(self.model, "get_language_model", None)
        return getter() if getter is not None else self.model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.sample(*args, **kwargs)

    def make_omni_output(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.make_omni_output(*args, **kwargs)

    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.preprocess(*args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.postprocess(*args, **kwargs)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loaded = self.model.load_weights(weights)
        return {f"model.{name}" for name in loaded}

    def embed_multimodal(self, **kwargs: Any) -> Any:
        return self.model.embed_multimodal(**kwargs)

    def embed_input_ids(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.embed_input_ids(*args, **kwargs)
