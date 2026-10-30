# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unified worker entry for Kimi-Audio's AR and acoustic stages."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from vllm.distributed import get_pp_group
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .processor import KimiAudioDummyInputsBuilder, KimiAudioMultiModalProcessor, KimiAudioProcessingInfo

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.sample.metadata import SamplingMetadata

    from vllm_omni.model_executor.models.output_templates import OmniOutput


@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor, info=KimiAudioProcessingInfo, dummy_inputs=KimiAudioDummyInputsBuilder
)
class KimiAudioForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """Construct only this worker's stage and expose its framework interfaces.

    The pipeline owns stage ordering and transport. The selected model owns
    computation, request hooks and checkpoint loading; this entry adds no
    generation loop or request state. Both stages use Kimi's root HF config.
    """

    have_multimodal_outputs = True
    has_postprocess = False

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = "") -> None:
        super().__init__()
        from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage
        model_prefix = maybe_prefix(prefix, "model")

        if self.model_stage == "kimi_audio_ar":
            if not get_pp_group().is_first_rank:
                # Only the input rank owns audio encoders. Set the worker's
                # config before registry construction copies it for the child.
                vllm_config.model_config.get_multimodal_config().skip_mm_profiling = True
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=model_prefix,
                hf_config=self.config,
                architectures=["KimiAudioARStage"],
            )
            # Preserve the existing AR runner contract without copying its
            # sampling context or implementing another sampler in this entry.
            for name in (
                "prefer_model_sampler",
                "requires_request_sample_eligibility",
                "skips_model_sampler_output_token_history",
                "omni_pooler_payload_include_hidden",
                "omni_client_multimodal_output_keys",
                "requires_full_prefix_cached_hidden_states",
            ):
                setattr(self, name, getattr(self.model, name))
            self.preprocess = self.model.preprocess
            self.preprocess_batch = self.model.preprocess_batch
            self.make_omni_output = self.model.make_omni_output
            self.sample = self.model.sample
        elif self.model_stage == "kimi_audio_decoder":
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=model_prefix,
                hf_config=self.config,
                architectures=["KimiAudioDecoder"],
            )
            self.enable_update_additional_information = self.model.enable_update_additional_information
            self.requires_raw_input_tokens = self.model.requires_raw_input_tokens
            self.requires_request_ids = self.model.requires_request_ids
            self.on_requests_finished = self.model.on_requests_finished
        else:
            raise ValueError(f"Unsupported Kimi-Audio model_stage: {self.model_stage!r}")

        self.has_preprocess = self.model.has_preprocess

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> None:
        # The message builder supplies token spans, not textual placeholders.
        return None

    def get_language_model(self) -> nn.Module:
        return self.model

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> "IntermediateTensors":
        # The acoustic stage still rejects PP > 1 during construction.
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def embed_multimodal(self, **kwargs: Any):
        if self.model_stage == "kimi_audio_ar":
            return self.model.embed_multimodal(**kwargs)
        return []

    def embed_input_ids(
        self, input_ids: torch.Tensor, multimodal_embeddings=None, *, is_multimodal: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.model_stage == "kimi_audio_ar":
            return self.model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal)
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: "IntermediateTensors | None" = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> "torch.Tensor | OmniOutput | tuple[torch.Tensor, ...]":
        output = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if isinstance(output, IntermediateTensors):
            # The existing Omni warmup extracts the first tensor of a tuple.
            # Keep every PP carrier as a flat tensor output, including under
            # CUDA-graph replay. make_omni_output restores the native carrier
            # outside forward, only for actual request execution.
            return tuple(output.tensors.values())
        return output

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: "SamplingMetadata | None" = None
    ) -> torch.Tensor | None:
        if self.model_stage == "kimi_audio_ar":
            return self.model.compute_logits(hidden_states, sampling_metadata)
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Delegate the lazy iterator unchanged: the acoustic stage loads its
        # own files and must never consume the root LLM checkpoint iterator.
        # Returned names are relative to this wrapper for loader validation.
        return {f"model.{name}" for name in self.model.load_weights(weights)}
