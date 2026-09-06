# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stateless multimodal preprocessing for VibeVoice reference audio."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import BatchFeature, PreTrainedConfig
from transformers.models.vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerFeatureExtractor
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import ModalityData, MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.inputs import AudioItem, MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from .vllm_compat import get_audio_with_sr_from_parent

AUDIO_BOS_TOKEN = "<|vision_start|>"
AUDIO_EOS_TOKEN = "<|vision_end|>"
AUDIO_TOKEN = "<|vision_pad|>"

SAMPLE_RATE = 24_000
AUDIO_HOP_LENGTH = 3_200
MAX_AUDIO_SECONDS = 60
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
MAX_AUDIO_TOKENS = math.ceil(MAX_AUDIO_SAMPLES / AUDIO_HOP_LENGTH)
MAX_AUDIO_ITEMS = 8

logger = init_logger(__name__)


class VibeVoiceMultiModalDataParser(MultiModalDataParser):
    """Reject ambiguous batches and make implicit downmixing visible."""

    def _parse_audio_data(
        self,
        data: ModalityData[AudioItem],
    ) -> MultiModalDataItems[Any, Any] | None:
        # Upstream interprets a bare 2D array as a batch of mono waveforms,
        # while callers often intend it to be one stereo waveform. Refuse to
        # guess: a tuple carries sample-rate and single-item intent explicitly;
        # a list carries batch intent explicitly.
        if isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 2:
            raise ValueError(
                "Ambiguous bare 2D VibeVoice audio input. Pass one multi-channel "
                "waveform as `(waveform, sample_rate)`, or pass multiple mono "
                "waveforms as a list."
            )
        return super()._parse_audio_data(data)

    def _get_audio_with_sr(
        self,
        audio: AudioItem,
    ) -> tuple[np.ndarray, float | None]:
        waveform, sample_rate = get_audio_with_sr_from_parent(super(), audio)
        if waveform.ndim > 1:
            logger.warning_once("Stereo or multi-channel VibeVoice reference audio is automatically downmixed to mono.")
        return waveform, sample_rate


class VibeVoiceProcessingInfo(BaseProcessingInfo):
    """Model and profiling information required by vLLM's MM processor."""

    def get_hf_config(self) -> PreTrainedConfig:
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        # The upstream VibeVoice Processor stores request-derived token counts
        # on ``self``. Keep the serving processor stateless and use only its
        # stateless Acoustic Tokenizer feature extractor.
        return None

    def get_feature_extractor(self, **kwargs: object) -> VibeVoiceAcousticTokenizerFeatureExtractor:
        return VibeVoiceAcousticTokenizerFeatureExtractor(
            feature_size=1,
            sampling_rate=SAMPLE_RATE,
            padding_value=0.0,
            normalize_audio=True,
            target_dB_FS=-25,
            eps=1e-6,
        )

    def get_data_parser(self) -> MultiModalDataParser:
        # MultiModalDataParser applies resampling first and channel
        # normalization second. The feature extractor then performs dB
        # normalization and 3200-sample padding on mono 24 kHz waveforms.
        return VibeVoiceMultiModalDataParser(
            target_sr=SAMPLE_RATE,
            target_channels=1,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": MAX_AUDIO_ITEMS}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        return {"audio": MAX_AUDIO_TOKENS}

    def audio_token_ids(self) -> tuple[int, int, int]:
        config = self.get_hf_config()
        return (
            int(config.audio_bos_token_id),
            int(config.audio_token_id),
            int(config.audio_eos_token_id),
        )


class VibeVoiceDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceProcessingInfo]):
    """Build worst-case reference-audio inputs for memory profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        segment = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
        return " ".join(segment for _ in range(mm_counts.get("audio", 0)))

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        overrides = mm_options.get("audio") if mm_options else None
        audios = self._get_dummy_audios(
            length=MAX_AUDIO_SAMPLES,
            num_audios=num_audios,
            overrides=overrides,
        )
        return {"audio": [(audio, SAMPLE_RATE) for audio in audios]}


class VibeVoiceMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceProcessingInfo]):
    """Tokenize prompts and expand each reference-audio placeholder."""

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # ``_call_hf_processor`` deliberately leaves one AUDIO_TOKEN per item;
        # vLLM applies PromptReplacement exactly once after audio processing.
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt

        audios = mm_data.get("audios")
        if audios is None or (isinstance(audios, (list, tuple)) and not audios):
            audios = mm_data.get("audio", [])
        _, audio_token_id, _ = self.info.audio_token_ids()
        num_placeholders = sum(int(token_id) == audio_token_id for token_id in prompt_ids)
        num_audios = len(audios)
        if num_placeholders != num_audios:
            # Match vLLM's own MM placeholder validation terminology. The
            # framework catches missing placeholders, but without this check an
            # extra AUDIO_TOKEN can remain as an ordinary text embedding.
            raise RuntimeError(
                f"Expected there to be {num_audios} prompt placeholders "
                f"corresponding to {num_audios} audio items, but instead found "
                f"{num_placeholders} prompt placeholders!"
            )
        if num_audios == 0:
            return BatchFeature({"input_ids": [prompt_ids]}, tensor_type="pt")

        raw_audios: list[object] = []
        for item_idx, item in enumerate(audios):
            audio = item[0] if isinstance(item, tuple) else item
            num_samples = len(audio)
            if num_samples <= 0:
                raise ValueError(f"VibeVoice audio item {item_idx} is empty.")
            if num_samples > MAX_AUDIO_SAMPLES:
                duration = num_samples / SAMPLE_RATE
                raise ValueError(
                    f"VibeVoice audio item {item_idx} is {duration:.2f}s; the maximum is {MAX_AUDIO_SECONDS}s."
                )
            raw_audios.append(audio)

        features = self.info.get_feature_extractor()(
            raw_audios,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            pad_to_multiple_of=AUDIO_HOP_LENGTH,
            return_attention_mask=True,
            return_tensors="pt",
        )
        padding_mask = features["padding_mask"].to(torch.long)
        valid_samples = padding_mask.sum(dim=-1)
        audio_num_tokens = torch.div(
            valid_samples + AUDIO_HOP_LENGTH - 1,
            AUDIO_HOP_LENGTH,
            rounding_mode="floor",
        ).to(torch.long)

        return BatchFeature(
            {
                "input_ids": [prompt_ids],
                "input_values": features["input_values"],
                "padding_mask": padding_mask,
                "audio_num_tokens": audio_num_tokens,
            },
            tensor_type=None,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "input_values": MultiModalFieldConfig.batched("audio"),
            "padding_mask": MultiModalFieldConfig.batched("audio"),
            "audio_num_tokens": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        _, audio_token_id, _ = self.info.audio_token_ids()
        out_mm_data = out_mm_kwargs.get_data()
        counts = out_mm_data.get("audio_num_tokens")
        if counts is None:
            counts_list = [MAX_AUDIO_TOKENS] * mm_items.get_count("audio")
        elif isinstance(counts, torch.Tensor):
            counts_list = counts.flatten().tolist()
        else:
            counts_list = list(counts)

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            num_tokens = int(counts_list[item_idx])
            if num_tokens < 1:
                raise ValueError(f"VibeVoice audio item {item_idx} produced no placeholder tokens.")
            return PromptUpdateDetails.select_token_id(
                [audio_token_id] * num_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_replacement,
            )
        ]


__all__ = [
    "AUDIO_BOS_TOKEN",
    "AUDIO_EOS_TOKEN",
    "AUDIO_HOP_LENGTH",
    "AUDIO_TOKEN",
    "MAX_AUDIO_ITEMS",
    "MAX_AUDIO_SECONDS",
    "MAX_AUDIO_TOKENS",
    "SAMPLE_RATE",
    "VibeVoiceDummyInputsBuilder",
    "VibeVoiceMultiModalDataParser",
    "VibeVoiceMultiModalProcessor",
    "VibeVoiceProcessingInfo",
]
