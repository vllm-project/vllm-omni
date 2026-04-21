from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.minicpmo import (
    MiniCPMODummyInputsBuilder,
    MiniCPMOMultiModalProcessor,
    MiniCPMOProcessingInfo,
)
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMOMultiModalProcessor,
    info=MiniCPMOProcessingInfo,
    dummy_inputs=MiniCPMODummyInputsBuilder,
)
class MiniCPMO4_5ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
):
    """MiniCPM-o 4.5 stage wrapper for thinker / talker / code2wav."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.model_stage = vllm_config.model_config.model_stage

        self.thinker = None
        self.talker = None
        self.code2wav = None
        self.model = None

        if self.model_stage == "thinker":
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config,
                architectures=["MiniCPMO4_5ThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        elif self.model_stage == "talker":
            self.talker = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config,
                architectures=["MiniCPMO4_5TalkerForConditionalGeneration"],
            )
            self.model = self.talker
            self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        elif self.model_stage == "code2wav":
            self.code2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config,
                architectures=["MiniCPMO4_5Code2Wav"],
            )
            self.model = self.code2wav

        if self.model is None:
            raise ValueError(f"Unsupported MiniCPMO4_5 model_stage: {self.model_stage}")

        self.has_preprocess = bool(getattr(self.model, "has_preprocess", False))
        self.has_postprocess = bool(getattr(self.model, "has_postprocess", False))
        self.requires_raw_input_tokens = bool(getattr(self.model, "requires_raw_input_tokens", False))
        self.enable_update_additional_information = bool(
            getattr(self.model, "enable_update_additional_information", False)
        )
        self.gpu_resident_buffer_keys = set(getattr(self.model, "gpu_resident_buffer_keys", set()))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = set()

        if self.thinker:
            thinker_loaded = self.thinker.load_weights(weights)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        if self.talker:
            talker_loaded = self.talker.load_weights(weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)

        if self.code2wav:
            code2wav_loaded = self.code2wav.load_weights(weights)
            code2wav_loaded = add_prefix_to_loaded_weights(code2wav_loaded, "code2wav")
            loaded_weights.update(code2wav_loaded)

        logger.info(
            "Loaded %d weights for MiniCPMO4_5 (stage=%s)",
            len(loaded_weights),
            self.model_stage,
        )
        return loaded_weights

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

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **info_dict: Any,
    ):
        if not self.has_preprocess:
            return input_ids, input_embeds, {}
        return self.model.preprocess(input_ids=input_ids, input_embeds=input_embeds, **info_dict)

    def postprocess(self, hidden_states: torch.Tensor, **info_dict: Any):
        if not self.has_postprocess:
            return {}
        return self.model.postprocess(hidden_states, **info_dict)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if hasattr(self.model, "make_omni_output"):
            return self.model.make_omni_output(model_outputs, **kwargs)
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        generate_audio: bool = True,
        voice_type: str = "ethan",
        codec: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            generate_audio=generate_audio,
            voice_type=voice_type,
            codec=codec,
            sampling_metadata=sampling_metadata,
            logits_index=logits_index,
            runtime_additional_information=runtime_additional_information,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self.model.compute_logits(hidden_states)
