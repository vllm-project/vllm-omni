# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native multimodal preprocessing for Ming-3.0-audio-flash."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import BatchFeature
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
)

from vllm_omni.inputs.mm_processor import OmniMultiModalProcessor
from vllm_omni.model_executor.models.common.ming.audio_lengths import (
    max_mel_frames_for_audio_tokens,
    ming3_audio_token_count,
)
from vllm_omni.transformers_utils.configs.ming3_audio import (
    BailingMoeV3AudioConfig,
)
from vllm_omni.transformers_utils.processors.ming import (
    MingWhisperFeatureExtractor,
)

AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_PAD_TOKEN = "<|audio_pad|>"
AUDIO_END_TOKEN = "<|audio_end|>"
ASR_PROMPT = (
    "Please recognize the language of this speech and transcribe it with "
    "appropriate punctuation marks, "
    "placing a `\t` between the identified language and the transcribe."
)
_SAMPLE_RATE = 16000
_SAMPLES_PER_MEL_FRAME = 160


def _audio_config_args(config: BailingMoeV3AudioConfig) -> dict[str, object]:
    audio_config = config.audio_config
    return {
        "chunk_size": audio_config.whisper_chunk_size,
        "kernels": audio_config.downsampler_kernel_size_list,
        "strides": audio_config.downsampler_stride_list,
        "paddings": audio_config.downsampler_padding_list,
    }


def get_ming3_audio_placeholder(modality: str, index: int) -> str:
    del index
    if modality.startswith("audio"):
        return f"{AUDIO_START_TOKEN}{AUDIO_PAD_TOKEN}{AUDIO_END_TOKEN}"
    raise ValueError("Ming-3.0-audio-flash only supports audio inputs")


class Ming3AudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> BailingMoeV3AudioConfig:
        return self.ctx.get_hf_config(BailingMoeV3AudioConfig)

    def get_hf_processor(self, **kwargs: object) -> None:
        return None

    def get_feature_extractor(self, **kwargs: object) -> MingWhisperFeatureExtractor:
        del kwargs
        config = self.get_hf_config().audio_config
        return MingWhisperFeatureExtractor(
            feature_size=config.n_mels,
            sampling_rate=_SAMPLE_RATE,
        )

    def get_data_parser(self) -> MultiModalDataParser:
        # AudioSpec's default downmix strategy is channel mean, matching the
        # released processor, and the parser performs the 16 kHz resampling.
        return MultiModalDataParser(target_sr=_SAMPLE_RATE, target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # The deployment decides the item limit, including historical audio.
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        num_audios = max(1, mm_counts.get("audio", 0))
        non_audio_prompt = f"{AUDIO_START_TOKEN}{AUDIO_END_TOKEN}" * num_audios + ASR_PROMPT
        prompt_tokens = len(self.get_tokenizer().encode(non_audio_prompt))
        # All items share one context. Reserving the whole sequence for each
        # item overestimates encoder profiling memory for multi-turn histories.
        token_budget = (seq_len - prompt_tokens) // num_audios
        if token_budget < 1:
            raise ValueError("Context length is too small for the configured audio item count")
        return {"audio": token_budget}


class Ming3AudioDummyInputsBuilder(BaseDummyInputsBuilder[Ming3AudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return get_ming3_audio_placeholder("audio", 0) * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        if not num_audios:
            return {}
        config = self.info.get_hf_config()
        token_budget = self.info.get_mm_max_tokens_per_item(seq_len, mm_counts)["audio"]
        mel_frames = max_mel_frames_for_audio_tokens(token_budget, **_audio_config_args(config))
        sample_count = max(1, mel_frames * _SAMPLES_PER_MEL_FRAME)
        return {
            "audio": self._get_dummy_audios(
                length=sample_count,
                num_audios=num_audios,
                overrides=(mm_options or {}).get("audio"),
            )
        }


class Ming3AudioMultiModalProcessor(OmniMultiModalProcessor[Ming3AudioProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _call_hf_processor(
        self,
        prompt: str | list[int],
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        del mm_kwargs, tok_kwargs
        prompt_ids = self.info.get_tokenizer().encode(prompt) if isinstance(prompt, str) else prompt
        audios = mm_data.get("audios", []) or mm_data.get("audio", [])
        if not audios:
            return BatchFeature({"input_ids": torch.tensor([prompt_ids], dtype=torch.long)})
        # The shared Ming extractor packs clips along the time dimension.
        # Extract each item independently so vLLM can cache and replace every
        # message placeholder separately, even for unequal or repeated clips.
        extractor = self.info.get_feature_extractor()
        audio_feats = []
        audio_lengths = []
        for audio in audios:
            if isinstance(audio, tuple):
                waveform, sample_rate = audio
            else:
                waveform, sample_rate = audio, _SAMPLE_RATE
            features = extractor([(np.asarray(waveform), sample_rate)], return_tensors="pt")
            audio_feats.append(features["audio_feats"].squeeze(0))
            audio_lengths.append(features["audio_feats_lengths"].reshape(1))
        return BatchFeature(
            {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
                # Preserve the existing tensor layout for a single clip.
                "audio_feats": audio_feats[0].unsqueeze(0) if len(audio_feats) == 1 else audio_feats,
                "audio_feats_lengths": torch.stack(audio_lengths),
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        return {
            "audio_feats": MultiModalFieldConfig.batched("audio"),
            "audio_feats_lengths": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        del mm_items, hf_processor_mm_kwargs
        tokenizer = self.info.get_tokenizer()
        audio_token_id = tokenizer.get_vocab()[AUDIO_PAD_TOKEN]
        lengths = out_mm_kwargs.get_data().get("audio_feats_lengths")
        if lengths is None:
            token_counts: list[int] = []
        else:
            token_counts = [
                ming3_audio_token_count(
                    int(length),
                    **_audio_config_args(self.info.get_hf_config()),
                )
                for length in torch.as_tensor(lengths).flatten()
            ]

        def replacement(item_idx: int) -> PromptUpdateDetails:
            count = token_counts[item_idx]
            if count <= 0:
                raise ValueError("Audio is too short to produce a model token")
            return PromptUpdateDetails.select_token_id(
                [audio_token_id] * count,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=replacement,
            )
        ]
