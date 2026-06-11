# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.minimind_o.minimind_omni_code2wav import (
    MiniMindOmniCode2Wav,
)
from vllm_omni.model_executor.models.minimind_o.minimind_omni_thinker import (
    MiniMindOmniDummyInputsBuilder,
    MiniMindOmniMultiModalProcessor,
    MiniMindOmniProcessingInfo,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


@MULTIMODAL_REGISTRY.register_processor(
    MiniMindOmniMultiModalProcessor,
    info=MiniMindOmniProcessingInfo,
    dummy_inputs=MiniMindOmniDummyInputsBuilder,
)
class MiniMindOmniForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.model_stage = vllm_config.model_config.model_stage
        self.requires_raw_input_tokens = False

        if self.model_stage == "thinker":
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=self.config,
                architectures=["MiniMindOmniThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.talker = None
            self.code2wav = None
        elif self.model_stage == "talker":
            self.multimodal_config.skip_mm_profiling = True
            self.has_preprocess = True
            self.has_postprocess = True
            self.thinker = None
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=self.config,
                architectures=["MiniMindOmniTalkerForConditionalGeneration"],
            )
            self.model = self.talker
            self.code2wav = None
            self.requires_raw_input_tokens = True
            self.mtp_hidden_size = self.talker.mtp_hidden_size
            self.talker_mtp_output_key = self.talker.talker_mtp_output_key
            self.gpu_resident_buffer_keys = self.talker.gpu_resident_buffer_keys
            self.talker_mtp_graph_safe = False
        elif self.model_stage == "code2wav":
            self.multimodal_config.skip_mm_profiling = True
            self.thinker = None
            self.talker = None
            self.code2wav = MiniMindOmniCode2Wav(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
            )
            self.model = self.code2wav
            self.requires_raw_input_tokens = True
        else:
            raise ValueError(
                f"Invalid MiniMind-Omni model_stage: {self.model_stage!r}. "
                "Expected one of: 'thinker', 'talker', 'code2wav'."
            )

        self.have_multimodal_outputs = getattr(self.model, "have_multimodal_outputs", False)
        self.prefer_model_sampler = getattr(self.model, "prefer_model_sampler", False)
        self.make_empty_intermediate_tensors = getattr(self.model, "make_empty_intermediate_tensors", lambda: None)

    def get_language_model(self) -> nn.Module:
        return self.model.get_language_model()

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        return self.model.embed_multimodal(**kwargs)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if self.model_stage == "thinker":
            kwargs.setdefault("return_hidden_states", True)
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ):
        return self.model.preprocess(input_ids, input_embeds, **info_dict)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if self.model_stage == "thinker":
            hidden_states, captured = model_outputs
            mm_outputs = captured if captured is not None else {}
            return OmniOutput(
                text_hidden_states=hidden_states.reshape(-1, hidden_states.shape[-1]),
                multimodal_outputs=mm_outputs,
            )
        if hasattr(self.model, "make_omni_output"):
            return self.model.make_omni_output(model_outputs, **kwargs)
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, **kwargs: Any) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, **kwargs)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.model.sample(logits, sampling_metadata)

    def postprocess(self, hidden_states: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        return self.model.postprocess(hidden_states, **kwargs)

    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        **kwargs: Any,
    ):
        return self.model.talker_mtp(
            input_ids,
            input_embeds,
            last_talker_hidden,
            text_step,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = self.model.load_weights(weights)
        if self.model_stage in {"thinker", "talker"}:
            return {f"{self.model_stage}.{name}" for name in loaded}
        return loaded
