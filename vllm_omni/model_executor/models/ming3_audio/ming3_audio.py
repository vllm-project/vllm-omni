# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only Ming-3.0-audio-flash model."""

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import PromptType, TokensPrompt
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    StreamingTranscriptionPostProcessor,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.model_executor.models.common.ming.audio_lengths import (
    iter_chunk_lengths,
)
from vllm_omni.model_executor.models.common.ming.audio_projector import (
    Ming3AudioProjector,
)
from vllm_omni.model_executor.models.common.ming.whisper_encoder import (
    WhisperAudioEncoder,
)
from vllm_omni.transformers_utils.configs.ming3_audio import (
    BailingMoeV3AudioConfig,
)

from .processing import (
    ASR_PROMPT,
    Ming3AudioDummyInputsBuilder,
    Ming3AudioMultiModalProcessor,
    Ming3AudioProcessingInfo,
    get_ming3_audio_placeholder,
)

_SAMPLE_RATE = 16000
_WHISPER_ENCODER_BATCH_SIZE = 32


class Ming3AudioStreamingPostProcessor(StreamingTranscriptionPostProcessor):
    """Hide the leading language field emitted before the first tab."""

    def __init__(self) -> None:
        self.raw_text = ""
        self.emitted_text = ""

    def process_delta(self, text_delta: str, finished: bool) -> str:
        self.raw_text += text_delta
        if "\t" not in self.raw_text:
            if not finished:
                return ""
            processed = self.raw_text
        else:
            processed = self.raw_text.split("\t", 1)[1]
        delta = processed[len(self.emitted_text) :]
        self.emitted_text = processed
        return delta


@MULTIMODAL_REGISTRY.register_processor(
    Ming3AudioMultiModalProcessor,
    info=Ming3AudioProcessingInfo,
    dummy_inputs=Ming3AudioDummyInputsBuilder,
)
class BailingMoeV3AudioForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
    HasInnerState,
    IsHybrid,
):
    """Whisper encoder + projector + native vLLM Bailing-MoE-v3 decoder."""

    # Downstream speech synthesis consumes generated text tokens only. Native
    # hybrid KV/recurrent-state caching needs no Omni prefix-output replay.
    requires_full_prefix_cached_hidden_states = False
    uses_native_kv_cache_only = True
    omni_pooler_payload_include_hidden = False

    supported_languages: Mapping[str, str] = {}
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": None,
            "model.audio.": "audio_encoder.",
            "linear_proj_audio.": "audio_projector.proj.",
            "linear_proj.": None,
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, index: int) -> str:
        return get_ming3_audio_placeholder(modality, index)

    @staticmethod
    def _text_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
        root_config = vllm_config.model_config.hf_config
        return vllm_config.with_hf_config(
            root_config.text_config,
            architectures=["BailingMoeV3ForCausalLM"],
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: BailingMoeV3AudioConfig = vllm_config.model_config.hf_config
        audio_config = self.config.audio_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = WhisperAudioEncoder(
                n_mels=audio_config.n_mels,
                n_ctx=audio_config.n_audio_ctx,
                n_state=audio_config.n_audio_state,
                n_head=audio_config.n_audio_head,
                n_layer=audio_config.n_audio_layer,
                learned_positional_embedding=True,
            )
            self.audio_projector = Ming3AudioProjector(
                audio_dim=audio_config.n_audio_state,
                llm_dim=self.config.text_config.hidden_size,
                kernels=audio_config.downsampler_kernel_size_list,
                strides=audio_config.downsampler_stride_list,
                paddings=audio_config.downsampler_padding_list,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = BailingMoeV3ForCausalLM(
                vllm_config=self._text_vllm_config(vllm_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _encode_audio_item(
        self,
        features: torch.Tensor,
        feature_length: int,
    ) -> torch.Tensor:
        features = features.reshape(-1, features.shape[-1])[:feature_length]
        chunk_lengths = iter_chunk_lengths(feature_length, self.config.audio_config.whisper_chunk_size)
        chunks = list(features.split(chunk_lengths))
        padded_length = max(chunk_lengths)
        embedding_batches = []
        for start in range(0, len(chunks), _WHISPER_ENCODER_BATCH_SIZE):
            batch_lengths = list(chunk_lengths[start : start + _WHISPER_ENCODER_BATCH_SIZE])
            chunk_batch = nn.utils.rnn.pad_sequence(
                chunks[start : start + _WHISPER_ENCODER_BATCH_SIZE], batch_first=True
            )
            # The unmasked encoder must retain the original global padding,
            # even when the final microbatch contains only a short tail.
            if chunk_batch.shape[1] < padded_length:
                chunk_batch = F.pad(chunk_batch, (0, 0, 0, padded_length - chunk_batch.shape[1]))
            encoded, encoded_lengths = self.audio_encoder.forward_padded(chunk_batch, batch_lengths)
            projected = self.audio_projector(encoded)
            projected_lengths = [self.audio_projector.output_length(length) for length in encoded_lengths]
            # Compact valid outputs so views do not retain padded activations.
            embedding_batches.append(
                torch.cat([projected[index, :length] for index, length in enumerate(projected_lengths)], dim=0)
            )
            del encoded, projected
        embeddings = embedding_batches[0] if len(embedding_batches) == 1 else torch.cat(embedding_batches, dim=0)
        if self.config.norm_query_embeds:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    @staticmethod
    def _normalize_audio_inputs(
        audio_feats: object,
        audio_feats_lengths: object,
    ) -> tuple[list[torch.Tensor], list[int]]:
        lengths = [int(length) for length in torch.as_tensor(audio_feats_lengths).flatten()]
        if isinstance(audio_feats, (list, tuple)):
            features = [torch.as_tensor(item) for item in audio_feats]
            features = [item.squeeze(0) if item.ndim == 3 and item.shape[0] == 1 else item for item in features]
        else:
            tensor = torch.as_tensor(audio_feats)
            if tensor.ndim == 2:
                features = [tensor]
            elif tensor.ndim == 3:
                features = [item for item in tensor]
            elif tensor.ndim == 4 and tensor.shape[1] == 1:
                features = [item.squeeze(0) for item in tensor]
            else:
                raise ValueError("audio_feats must be [time, mel], [items, time, mel], or [items, 1, time, mel]")
        if len(features) != len(lengths):
            raise ValueError(f"Received {len(features)} audio feature tensors but {len(lengths)} lengths")
        for feature, length in zip(features, lengths, strict=True):
            if feature.ndim != 2:
                raise ValueError("Each audio feature tensor must have shape [time, mel]")
            if not 0 < length <= feature.shape[0]:
                raise ValueError(f"Audio feature length {length} must be positive and fit {feature.shape[0]} frames")
        return features, lengths

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_feats = kwargs.pop("audio_feats", None)
        audio_feats_lengths = kwargs.pop("audio_feats_lengths", None)
        if audio_feats is None:
            return ()
        if audio_feats_lengths is None:
            raise ValueError("audio_feats_lengths is required with audio_feats")
        features, lengths = self._normalize_audio_inputs(audio_feats, audio_feats_lengths)
        return tuple(
            self._encode_audio_item(feature, length) for feature, length in zip(features, lengths, strict=True)
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if not multimodal_embeddings:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            tower_model=["audio_encoder.", "audio_projector."],
        )

    def get_num_mm_encoder_tokens(self, num_audio_tokens: int) -> int:
        return num_audio_tokens

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int], tuple[int, int, int]]:
        return BailingMoeV3ForCausalLM.get_mamba_state_shape_from_config(cls._text_vllm_config(vllm_config))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        return BailingMoeV3ForCausalLM.get_mamba_state_dtype_from_config(cls._text_vllm_config(vllm_config))

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[Any, ...]:
        return BailingMoeV3ForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig, task_type: str) -> SpeechToTextConfig:
        # Upstream initializes both speech-to-text handlers for the shared transcription task.
        # Keep construction side-effect free and defer unsupported translation rejection to prompt construction.
        if task_type not in {"transcribe", "translate"}:
            raise ValueError(f"Unsupported speech-to-text task: {task_type}")
        return SpeechToTextConfig(
            sample_rate=_SAMPLE_RATE,
            max_audio_clip_s=None,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        if stt_params.task_type != "transcribe":
            raise ValueError("Ming-3.0-audio-flash does not support translation")
        if stt_params.request_prompt:
            raise ValueError("Ming-3.0-audio-flash does not support a custom ASR prompt")
        tokenizer = cached_tokenizer_from_config(stt_params.model_config)
        prompt_token_ids = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ASR_PROMPT},
                        {"type": "audio"},
                    ],
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data={"audio": stt_params.audio},
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return text.split("\t", 1)[1] if "\t" in text else text

    @classmethod
    def get_streaming_post_processor_cls(
        cls,
    ) -> type[StreamingTranscriptionPostProcessor]:
        return Ming3AudioStreamingPostProcessor
