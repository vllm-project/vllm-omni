# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio-7B unified dispatcher: routes to fused_thinker or code2wav
based on ``vllm_config.model_config.model_stage``."""

from collections.abc import Iterable
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.kimi_audio import (
    KimiAudioDummyInputsBuilder,
    KimiAudioProcessingInfo,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.kimi_audio.multimodal_processor import (
    OmniKimiAudioMultiModalProcessor,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    OmniKimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder,
)
class KimiAudioForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "fused_thinker":
            self.fused_thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "fused_thinker"),
                architectures=["KimiAudioThinkerForConditionalGeneration"],
            )
            self.code2wav = None
            self.model = self.fused_thinker
            # DefaultModelLoader reads ``secondary_weights`` off the top-level
            # module — forward the fused thinker's whisper-large-v3 entry
            # so encoder weights reach load_weights.
            inner_secondary = getattr(self.fused_thinker, "secondary_weights", None)
            if inner_secondary:
                self.secondary_weights = inner_secondary
        elif self.model_stage == "code2wav":
            self.fused_thinker = None
            self.code2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                architectures=["KimiAudioCode2Wav"],
            )
            self.model = self.code2wav
            self.requires_raw_input_tokens = True
        else:
            raise ValueError(f"Invalid model_stage: {self.model_stage}. Must be one of: 'fused_thinker', 'code2wav'.")

        self.make_empty_intermediate_tensors = (
            self.fused_thinker.make_empty_intermediate_tensors if self.model_stage == "fused_thinker" else lambda: None
        )

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def embed_multimodal(self, **kwargs):
        return self.model.embed_multimodal(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        dev = self._module_device(self.model)
        if input_ids is not None and input_ids.device != dev:
            input_ids = input_ids.to(dev)
        if positions is not None and positions.device != dev:
            positions = positions.to(dev)
        if inputs_embeds is not None and inputs_embeds.device != dev:
            inputs_embeds = inputs_embeds.to(dev)
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        text_hidden_states = model_outputs
        if isinstance(text_hidden_states, torch.Tensor):
            text_hidden_states = text_hidden_states.reshape(-1, text_hidden_states.shape[-1])
        return OmniOutput(text_hidden_states=text_hidden_states, multimodal_outputs=None)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        text_logits = self.model.compute_logits(hidden_states)
        if text_logits is None or self.model_stage != "fused_thinker":
            return text_logits
        if not getattr(self.fused_thinker, "_generate_audio", False):
            return text_logits

        # Audio-out mode terminates on the audio head, not the text head;
        # suppress text EOS until the MIMO head fires.
        text_logits[..., 151644] = float("-inf")
        text_logits[..., 151645] = float("-inf")
        # Suppress audio-stream control tokens — sampling one would
        # re-trigger the prefill-detection branch and clobber _req_state.
        for _ctrl in (151661, 151663, 151670, 151671, 151675, 151676):
            text_logits[..., _ctrl] = float("-inf")

        state = getattr(self.fused_thinker, "_req_state", {})
        if state.get("audio_eod_seen"):
            text_logits[..., 151644] = float("inf")
        elif state.get("text_eos_seen"):
            # Force kimia_text_blank to match the masked input the model
            # sees post-EOS, so the displayed transcript stays clean.
            text_logits[...] = float("-inf")
            text_logits[..., 151666] = 0.0
        return text_logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # code2wav loads its detokenizer/vocoder out-of-band (audio_detokenizer
        # from <model_path>/, BigVGAN from a separate HF repo) and ignores
        # anything routed through here.
        loaded: set[str] = set()
        sub_loaded = self.model.load_weights(weights)
        prefix = self.model_stage
        loaded.update(add_prefix_to_loaded_weights(sub_loaded, prefix))
        logger.info(
            "Loaded %d weights for KimiAudio (stage=%s)",
            len(loaded),
            self.model_stage,
        )
        return loaded
