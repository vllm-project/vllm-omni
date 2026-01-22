from collections.abc import Iterable
from functools import cached_property

import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTalkerConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder

# from vllm.attention import AttentionMetadata  # unused import
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import Qwen2_5OmniConditionalGenerationMixin


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniTalkerForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, Qwen2_5OmniConditionalGenerationMixin
):
    logger = init_logger(__name__)
    # Align to thinker-style static mapper for clarity
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # text LM head/body in talker
            "talker.codec_head.": "language_model.lm_head.",
            "talker.model.": "language_model.model.",
            # projection weights
            "talker.thinker_to_talker_proj.": "thinker_to_talker_proj.",
            # fallback root
            "talker.": "",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2_5OmniTalkerConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.quant_config = quant_config

        if hasattr(config, "talker_config"):
            self.config = config.talker_config
            vllm_config.model_config.hf_text_config = vllm_config.model_config.hf_config.talker_config
        else:
            self.config = config

        self.thinker_to_talker_proj = nn.Linear(
            self.config.embedding_size,
            self.config.hidden_size,
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=getattr(self.config, "text_config", self.config),
            architectures=["Qwen2ForCausalLM_old"],
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        # suppress start id
        self.suppress_start_id = None

    def init_multi_modal(self, thinker_config):
        self.audio_tower = Qwen2_5OmniAudioEncoder(thinker_config.audio_config)
        self.visual = Qwen2_5_VisionTransformer(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=self.quant_config,
            prefix=maybe_prefix(self.prefix, "visual"),
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return Sampler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        assert input_ids is not None or inputs_embeds is not None, "input_ids or inputs_embeds must be provided"
        # forward_context: ForwardContext = get_forward_context()  # unused variable

        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            # for profile_run:
            inputs_embeds = self.embed_input_ids(input_ids)

        input_ids = None

        # projection
        inputs_embeds = self.thinker_to_talker_proj(inputs_embeds)

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def bad_word_processor(self, logits: torch.Tensor) -> torch.Tensor:
        # suppress token IDs unsupported by token2wav
        if self.suppress_start_id and self.suppress_start_id < logits.size(-1):
            # skip the end token id.
            if hasattr(self.config, "tts_codec_end_token_id"):
                end_id = int(getattr(self.config, "tts_codec_end_token_id"))
                if self.suppress_start_id == end_id:
                    logits[..., end_id + 1 : logits.size(-1)] = -1e9
                elif self.suppress_start_id < end_id:
                    logits[..., self.suppress_start_id : end_id] = -1e9
                    logits[..., end_id + 1 : logits.size(-1)] = -1e9
                else:
                    logits[..., self.suppress_start_id : logits.size(-1)] = -1e9
            else:
                raise ValueError("config must have tts_codec_end_token_id attribute")

        if hasattr(self.config, "tts_codec_start_token_id"):
            bos_id = int(getattr(self.config, "tts_codec_start_token_id"))
            logits[..., bos_id] = -1e9
        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.language_model.compute_logits(hidden_states)
        logits = self.bad_word_processor(logits)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "token2wav."],
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            self.logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            pass
        multi_model_weights = set()
        for name, param in self.visual.named_parameters():
            multi_model_weights.add("visual." + name)
        for name, param in self.audio_tower.named_parameters():
            multi_model_weights.add("audio_tower." + name)
        loaded.update(multi_model_weights)
        return loaded

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(**kwargs)
            if input_key in ("input_audio_features") and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(**kwargs)
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += audio_embeddings
        return multimodal_embeddings

    def set_suppress_start_id(self, start_id: int):
        self.suppress_start_id = start_id
        self.logger.debug(f"Set suppress start id to {self.suppress_start_id}")
