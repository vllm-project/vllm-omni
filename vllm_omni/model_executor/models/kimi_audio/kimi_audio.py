# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unified worker entry for Kimi-Audio's AR and acoustic stages."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sequence import IntermediateTensors
    from vllm.v1.sample.metadata import SamplingMetadata

    from vllm_omni.model_executor.models.output_templates import OmniOutput


class KimiAudioForConditionalGeneration(nn.Module):
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
        else:
            raise ValueError(f"Unsupported Kimi-Audio model_stage: {self.model_stage!r}")

        self.has_preprocess = self.model.has_preprocess

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Framework embedding hook; AR's dual-stream fusion stays in preprocess."""
        if self.model_stage == "kimi_audio_ar":
            return self.model.embed_tokens(input_ids)
        return self.model.embed_input_ids(input_ids, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: "IntermediateTensors | None" = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> "torch.Tensor | OmniOutput":
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

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
