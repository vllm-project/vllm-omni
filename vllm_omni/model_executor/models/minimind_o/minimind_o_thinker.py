# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# MiniMind-O Thinker stage with frozen SenseVoice-Small and SigLIP2 encoders.
# Following vLLM standalone model pattern.

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
)

from vllm_omni.model_executor.models.minimind_o.config import MiniMindOThinkerConfig
from vllm_omni.model_executor.models.minimind_o.minimind_mm_utils import (
    inject_audio_features,
    inject_vision_features,
)
from vllm_omni.model_executor.models.minimind_o.projectors import (
    MiniMindOAudioProjector,
    MiniMindOVisionProjector,
)

logger = init_logger(__name__)


class MiniMindOThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsMRoPE,
    Qwen2_5OmniConditionalGenerationMixin,
):
    """
    MiniMind-O Thinker stage with frozen encoders.

    Components:
    - SenseVoice-Small encoder (frozen, loaded as audio_tower)
    - SigLIP2 encoder (frozen, loaded as visual)
    - Audio projector (2-layer MLP)
    - Vision projector (2-layer MLP)
    - LLM backbone (dense or MoE)
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "audio_proj.": "audio_tower.projector.",
            "vision_proj.": "visual.projector.",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image_pad|>"
        if modality.startswith("audio"):
            return "<|audio_pad|>"
        raise ValueError("Only image or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MiniMindOThinkerConfig = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.config = config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.multimodal_config = multimodal_config

        # Initialize audio tower (SenseVoice-Small encoder + projector)
        with self._mark_tower_model(vllm_config, "audio"):
            if multimodal_config.get_limit_per_prompt("audio"):
                self.audio_tower = nn.Module()
                self.audio_tower.encoder = None  # Will be loaded from HF weights
                self.audio_tower.projector = MiniMindOAudioProjector(
                    config.audio_hidden_size,
                    config.hidden_size,
                )
            else:
                self.audio_tower = None

        # Initialize visual tower (SigLIP2 encoder + projector)
        with self._mark_tower_model(vllm_config, {"image"}):
            if multimodal_config.get_limit_per_prompt("image"):
                self.visual = nn.Module()
                self.visual.encoder = None  # Will be loaded from HF weights
                self.visual.projector = MiniMindOVisionProjector(
                    config.image_hidden_size,
                    config.hidden_size,
                    target_tokens=config.image_token_len,
                )
            else:
                self.visual = None

        # Initialize LLM backbone (dense or MoE)
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
                hf_config=config.text_config,
                architectures=["MiniMindForCausalLM"],
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        # Special tokens
        self.audio_pad_token = config.audio_pad_token
        self.audio_stop_token = config.audio_stop_token
        self.audio_spk_token = config.audio_spk_token
        self.audio_ids = config.audio_ids
        self.image_ids = config.image_ids

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or is_multimodal is None:
            return self.language_model.embed_input_ids(input_ids)

        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if len(multimodal_embeddings) == 0:
            return inputs_embeds

        # Standard merge for multimodal embeddings
        return self._merge_multimodal_embeddings(
            inputs_embeds,
            multimodal_embeddings,
            is_multimodal,
            input_ids,
        )

    def _merge_multimodal_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_multimodal: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Merge multimodal embeddings into input embeddings."""
        # Simple merge: replace multimodal token positions with embeddings
        # This is a simplified version - full implementation would handle interleaving
        merged_embeds = inputs_embeds.clone()
        mm_idx = 0
        for i in range(input_ids.shape[1]):
            if is_multimodal[0, i]:
                if mm_idx < len(multimodal_embeddings):
                    merged_embeds[:, i] = multimodal_embeddings[mm_idx]
                    mm_idx += 1
        return merged_embeds

    def _maybe_inject_multimodal(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        if inputs_embeds is None or input_ids is None:
            return inputs_embeds
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        audio_inputs = kwargs.get("audio_inputs")
        audio_lens = kwargs.get("audio_lens")
        pixel_values = kwargs.get("pixel_values")

        hidden_states = inputs_embeds
        if audio_inputs is not None and self.audio_tower is not None and self.config.audio_ids:
            # Encoder is optional (loaded from funasr at runtime); skip if not present.
            if getattr(self.audio_tower, "encoder", None) is not None:
                hidden_states = inject_audio_features(
                    input_ids,
                    hidden_states,
                    self._encode_audio_inputs(audio_inputs, audio_lens),
                    audio_marker=self.config.audio_ids[0],
                )
        if pixel_values is not None and self.visual is not None and self.config.image_ids:
            vision_tensors = self._encode_image_inputs(pixel_values)
            if vision_tensors is not None:
                hidden_states = inject_vision_features(
                    input_ids,
                    hidden_states,
                    vision_tensors,
                    image_marker=self.config.image_ids[0],
                    seqlen=hidden_states.size(1),
                )
        return hidden_states.squeeze(0) if hidden_states.size(0) == 1 else hidden_states

    def _encode_audio_inputs(self, audio_inputs, audio_lens):
        return None

    def _encode_image_inputs(self, pixel_values):
        if self.visual is None or getattr(self.visual, "encoder", None) is None:
            return None
        mask = pixel_values.flatten(1).any(1)
        if not mask.any():
            return pixel_values.new_zeros(
                pixel_values.size(0),
                self.config.image_token_len,
                self.config.hidden_size,
            )
        with torch.no_grad():
            emb = self.visual.encoder(pixel_values=pixel_values[mask]).last_hidden_state
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        emb = self.visual.projector(emb)
        if mask.all():
            return emb
        idx = mask.nonzero().view(-1, 1, 1).expand_as(emb)
        return emb.new_zeros(pixel_values.size(0), *emb.shape[1:]).scatter(0, idx, emb)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_input_ids(input_ids)
        if inputs_embeds is not None and intermediate_tensors is None:
            inputs_embeds = self._maybe_inject_multimodal(input_ids, inputs_embeds, kwargs)

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["talker.", "code2wav."]
        if self.audio_tower is None:
            skip_prefixes.extend(["audio_tower."])
        if self.visual is None:
            skip_prefixes.extend(["visual."])

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        return loaded_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix in multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="",
            tower_model=["visual.", "audio_tower."],
        )
