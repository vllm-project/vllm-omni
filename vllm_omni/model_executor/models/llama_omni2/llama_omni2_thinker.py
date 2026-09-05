# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LLaMA-Omni 2 Thinker model components."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from transformers import BatchFeature
from transformers.processing_utils import ProcessorMixin
from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from vllm_omni.model_executor.models.output_templates import OmniOutput

SPEECH_TOKEN_INDEX = -200
SPEECH_TOKEN_ID = 151665

THINKER_WEIGHTS_MAPPER = Qwen2Model.hf_to_vllm_mapper | WeightsMapper(
    orig_to_new_prefix={
        "speech_generator.": None,
        "model.speech_encoder.": "speech_encoder.",
        "model.speech_projector.": "speech_projector.",
        "model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    }
)


class SpeechProjectorConfig(Protocol):
    speech_encoder_ds_rate: int
    speech_encoder_hidden_size: int
    hidden_size: int


class EncoderProjectorConcat(nn.Module):
    """Concatenate complete encoder frame groups before projection."""

    def __init__(self, config: SpeechProjectorConfig) -> None:
        super().__init__()
        self.k = int(config.speech_encoder_ds_rate)
        self.encoder_dim = int(config.speech_encoder_hidden_size)
        self.llm_dim = int(config.hidden_size)
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, encoder_dim = encoder_output.shape
        complete_length = sequence_length - sequence_length % self.k
        encoder_output = encoder_output[:, :complete_length].contiguous()
        grouped = encoder_output.view(
            batch_size,
            complete_length // self.k,
            encoder_dim * self.k,
        )
        return self.linear2(self.relu(self.linear1(grouped)))


def projected_speech_lengths(
    speech_lengths: torch.Tensor,
    *,
    projector_stride: int,
) -> torch.Tensor:
    """Convert input mel lengths to projected speech-token lengths."""
    if projector_stride <= 0:
        raise ValueError("projector_stride must be positive")
    whisper_lengths = (speech_lengths + 1) // 2
    return whisper_lengths // projector_stride


def speech_placeholder_token_ids(
    speech_lengths: torch.Tensor,
    *,
    projector_stride: int = 5,
) -> list[list[int]]:
    """Return one legal tokenizer placeholder per projected speech row."""
    output_lengths = projected_speech_lengths(
        speech_lengths,
        projector_stride=projector_stride,
    )
    return [[SPEECH_TOKEN_ID] * int(length) for length in output_lengths]


def splice_speech_embeddings(
    input_ids: torch.Tensor,
    speech_features: Sequence[torch.Tensor],
    *,
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    speech_token_index: int = SPEECH_TOKEN_INDEX,
) -> torch.Tensor:
    """Replace speech placeholders with their projected feature rows."""
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be one-dimensional, got shape {input_ids.shape}")

    placeholder_indices = torch.where(input_ids == speech_token_index)[0].tolist()
    if len(placeholder_indices) != len(speech_features):
        raise ValueError(
            f"speech placeholder count ({len(placeholder_indices)}) does not match "
            f"speech feature count ({len(speech_features)})"
        )

    boundaries = [-1, *placeholder_indices, input_ids.shape[0]]
    pieces: list[torch.Tensor] = []
    for feature_index in range(len(boundaries) - 1):
        text_ids = input_ids[boundaries[feature_index] + 1 : boundaries[feature_index + 1]]
        if text_ids.numel() > 0:
            pieces.append(embed_tokens(text_ids))
        if feature_index < len(speech_features):
            pieces.append(speech_features[feature_index])

    if pieces:
        return torch.cat(pieces, dim=0)
    return embed_tokens(input_ids)


def _import_openai_whisper():
    try:
        import whisper
    except ImportError as exc:
        raise ImportError("LLaMA-Omni 2 speech input requires the openai-whisper package") from exc
    return whisper


class LlamaOmni2Processor(ProcessorMixin):
    """Tokenize prompts and create OpenAI Whisper-compatible mel features."""

    attributes = ["tokenizer"]
    attribute_class = {"tokenizer": "PreTrainedTokenizerBase"}

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.audio_token = "<speech>"

    def __call__(
        self,
        text: str | list[str] | None = None,
        audio: object | Sequence[object] | None = None,
        **kwargs: object,
    ) -> BatchFeature:
        text = "" if text is None else text
        tokenized = dict(self.tokenizer(text, **kwargs))
        if audio is None:
            tokenized["speech"] = torch.empty(0, 128, 0)
            tokenized["speech_lengths"] = torch.empty(0, dtype=torch.long)
            return BatchFeature(tokenized)

        audio_items = list(audio) if isinstance(audio, (list, tuple)) else [audio]
        whisper = _import_openai_whisper()
        mel_features = []
        for item in audio_items:
            waveform = item
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.as_tensor(waveform, dtype=torch.float32)
            waveform = whisper.pad_or_trim(waveform.float())
            mel_features.append(whisper.log_mel_spectrogram(waveform, n_mels=128))

        tokenized["speech"] = torch.stack(mel_features)
        tokenized["speech_lengths"] = torch.tensor(
            [feature.shape[-1] for feature in mel_features],
            dtype=torch.long,
        )
        return BatchFeature(tokenized)


class LlamaOmni2ProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> LlamaOmni2Processor:
        del kwargs
        return LlamaOmni2Processor(self.get_tokenizer())

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000, target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        return {"audio": 300}


class LlamaOmni2DummyInputsBuilder(BaseDummyInputsBuilder[LlamaOmni2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<speech>" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        del seq_len
        overrides = None if mm_options is None else mm_options.get("audio")
        return {
            "audio": self._get_dummy_audios(
                length=480000,
                num_audios=mm_counts.get("audio", 0),
                overrides=overrides,
            )
        }


class LlamaOmni2MultiModalProcessor(BaseMultiModalProcessor[LlamaOmni2ProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        del prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = [item[0] if isinstance(item, tuple) else item for item in audios]
        output = self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )
        return BatchFeature(output)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        return {
            "speech": MultiModalFieldConfig.batched("audio"),
            "speech_lengths": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs
        speech_lengths = out_mm_kwargs.get_data().get("speech_lengths")
        if speech_lengths is None:
            output_lengths: list[list[int]] = []
        else:
            output_lengths = speech_placeholder_token_ids(torch.as_tensor(speech_lengths).flatten())

        def replacement(item_index: int) -> list[int]:
            if item_index >= len(output_lengths):
                return [SPEECH_TOKEN_ID] * 300
            return output_lengths[item_index]

        return [
            PromptReplacement(
                modality="audio",
                target="<speech>",
                replacement=replacement,
            )
        ]


def load_openai_whisper_encoder(model_name: str) -> nn.Module:
    """Construct the checkpoint-compatible OpenAI Whisper encoder."""
    whisper = _import_openai_whisper()
    WhisperLayerNorm = whisper.model.LayerNorm

    model_key = Path(model_name).stem
    if model_key != "large-v3":
        raise ValueError(f"LLaMA-Omni 2 requires a Whisper large-v3 speech encoder, got {model_name!r}")
    encoder = whisper.model.AudioEncoder(
        n_mels=128,
        n_ctx=1500,
        n_state=1280,
        n_head=20,
        n_layer=32,
    )

    def replace_layer_norm(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, WhisperLayerNorm):
                replacement = nn.LayerNorm(
                    child.normalized_shape,
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                )
                replacement.load_state_dict(child.state_dict())
                setattr(module, name, replacement)
            else:
                replace_layer_norm(child)

    replace_layer_norm(encoder)
    return encoder


@MULTIMODAL_REGISTRY.register_processor(
    LlamaOmni2MultiModalProcessor,
    info=LlamaOmni2ProcessingInfo,
    dummy_inputs=LlamaOmni2DummyInputsBuilder,
)
class LlamaOmni2ThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
):
    """Whisper speech tower composed with vLLM's native Qwen2 causal LM."""

    have_multimodal_outputs = True
    prefer_model_sampler = True
    has_postprocess = True
    cumulative_postprocess_output_buffer_keys = {
        ("ids", "output"),
        ("embed", "decode"),
        ("hidden_states", "output"),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        self.speech_encoder = load_openai_whisper_encoder(self.config.speech_encoder)
        self.speech_projector = EncoderProjectorConcat(self.config)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.config.thinker_config,
            architectures=["Qwen2ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality.startswith("audio"):
            return "<speech>"
        raise ValueError("Only audio modality is supported")

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def _process_speech_input(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if speech.ndim != 3:
            raise ValueError(f"speech must have shape (batch, mel_bins, frames), got {speech.shape}")
        if speech_lengths.ndim != 1 or speech_lengths.shape[0] != speech.shape[0]:
            raise ValueError("speech_lengths must contain one length for each speech item")
        if "whisper" not in self.config.speech_encoder_type.lower():
            raise ValueError(f"Unsupported speech encoder type: {self.config.speech_encoder_type}")

        encoder_output = self.speech_encoder(speech)
        projected = self.speech_projector(encoder_output)
        output_lengths = projected_speech_lengths(
            speech_lengths,
            projector_stride=self.speech_projector.k,
        )
        return tuple(projected[item_index, : int(item_length)] for item_index, item_length in enumerate(output_lengths))

    def embed_multimodal(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        speech = kwargs.get("speech")
        speech_lengths = kwargs.get("speech_lengths")
        if speech is None:
            return ()
        if not isinstance(speech, torch.Tensor):
            speech = torch.cat(list(speech), dim=0)
        if speech_lengths is None:
            raise ValueError("speech_lengths is required when speech is provided")
        if not isinstance(speech_lengths, torch.Tensor):
            speech_lengths = torch.cat(list(speech_lengths), dim=0)
        return self._process_speech_input(speech, speech_lengths)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor | object:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self.language_model.compute_logits(hidden_states)

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **_: object,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs={},
        )

    def sample(self, logits: torch.Tensor, sampling_metadata: object):
        sampler = getattr(self, "_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler

            sampler = Sampler()
            self._sampler = sampler
        output = sampler(logits=logits, sampling_metadata=sampling_metadata)
        self._last_sampled_token_ids = output.sampled_token_ids
        self._postprocess_cursor = 0
        return output

    def postprocess(
        self,
        hidden_states_slice: torch.Tensor,
        multimodal_outputs: object = None,
        **_: object,
    ) -> dict[str, dict[str, torch.Tensor]]:
        del multimodal_outputs
        sampled = getattr(self, "_last_sampled_token_ids", None)
        if not isinstance(sampled, torch.Tensor) or sampled.numel() == 0:
            return {}
        cursor = int(getattr(self, "_postprocess_cursor", 0))
        sampled = sampled.reshape(-1)
        if cursor >= sampled.shape[0] or hidden_states_slice.numel() == 0:
            return {}
        token_id = sampled[cursor : cursor + 1].to(
            device=hidden_states_slice.device,
            dtype=torch.long,
        )
        self._postprocess_cursor = cursor + 1
        return {
            "ids": {
                "output": token_id.detach(),
            },
            "embed": {
                "decode": self.language_model.embed_input_ids(token_id).detach(),
            },
            "hidden_states": {
                "output": hidden_states_slice[-1:].detach(),
            },
        }

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=THINKER_WEIGHTS_MAPPER)
