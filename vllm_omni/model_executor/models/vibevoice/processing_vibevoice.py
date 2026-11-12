# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stateless reference-audio processing for vLLM's data-only processor API.

Only the installed HF acoustic feature extractor is needed; the full HF TTS
processor is deliberately not instantiated. Token expansion depends solely on
this call's CPU metadata, including when multimodal items come from cache.
"""

from collections.abc import Mapping, Sequence
from functools import cached_property

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.vibevoice_acoustic_tokenizer.feature_extraction_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerFeatureExtractor,
)
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.audio import MONO_AUDIO_SPEC, normalize_audio
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.processing.processor import MultiModalProcessingResult, PlaceholderFeaturesInfo

from .prompt import AUDIO_BOS, AUDIO_EOS, AUDIO_TOKEN

SAMPLE_RATE = 24000
SAMPLES_PER_TOKEN = 3200
MAX_REFERENCE_SAMPLES = 60 * SAMPLE_RATE


class VibeVoiceDataParser(MultiModalDataParser):
    def _get_audio_with_sr(self, audio):
        # Upstream resamples before channel normalization, which resamples the
        # channel axis of (time, channels) audio; mix to 1-D mono first.
        waveform, sample_rate = super()._get_audio_with_sr(audio)
        if not isinstance(waveform, torch.Tensor):
            waveform = np.asarray(waveform)  # (list, sr) tuples are passed through as-is.
        return normalize_audio(waveform, MONO_AUDIO_SPEC).reshape(-1), sample_rate


class VibeVoiceProcessingInfo(BaseProcessingInfo):
    @cached_property
    def feature_extractor(self) -> VibeVoiceAcousticTokenizerFeatureExtractor:
        model_config = self.ctx.model_config
        extractor = VibeVoiceAcousticTokenizerFeatureExtractor.from_pretrained(
            model_config.model,
            revision=model_config.revision,
            trust_remote_code=False,
        )
        if extractor.sampling_rate != SAMPLE_RATE or extractor.pad_to_multiple_of != SAMPLES_PER_TOKEN:
            raise ValueError("VibeVoice requires the native 24 kHz / 3200-hop feature extractor.")
        return extractor

    def get_data_parser(self) -> MultiModalDataParser:
        return VibeVoiceDataParser(target_sr=SAMPLE_RATE, target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        return {"audio": MAX_REFERENCE_SAMPLES // SAMPLES_PER_TOKEN}


class VibeVoiceDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return (AUDIO_BOS + AUDIO_TOKEN + AUDIO_EOS) * mm_counts.get("audio", 0) + AUDIO_BOS

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        length = min(MAX_REFERENCE_SAMPLES, max(1, seq_len - 3) * SAMPLES_PER_TOKEN)
        return {"audio": self._get_dummy_audios(length=length, num_audios=mm_counts.get("audio", 0))}


class VibeVoiceMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceProcessingInfo]):
    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # The supported checkpoint fixes normalization, padding and sample
        # rate. Do not silently accept overrides that alter reference tokens.
        if hf_kwargs:
            raise ValueError("VibeVoice does not support mm_processor_kwargs overrides.")
        audio = hf_data["audio"]
        for item in audio:
            waveform = np.asarray(item)
            if waveform.ndim != 1 or not 0 < waveform.size <= MAX_REFERENCE_SAMPLES:
                raise ValueError("VibeVoice reference audio must be non-empty mono audio of at most 60 seconds.")
            if not np.isfinite(waveform).all():
                raise ValueError("VibeVoice reference audio must contain only finite samples.")
        outputs = self.info.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            pad_to_multiple_of=SAMPLES_PER_TOKEN,
            return_attention_mask=True,
            return_tensors="pt",
        )
        valid_samples = outputs.pop("padding_mask").sum(dim=-1)
        outputs["audio_num_tokens"] = torch.div(
            valid_samples + SAMPLES_PER_TOKEN - 1, SAMPLES_PER_TOKEN, rounding_mode="floor"
        ).to(torch.int64)
        return outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "input_values": MultiModalFieldConfig.batched("audio"),
            "audio_num_tokens": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        audio_token_id = self.info.get_hf_config().audio_token_id

        def replacement(item_idx: int) -> list[int]:
            count = int(out_mm_kwargs["audio"][item_idx]["audio_num_tokens"].data.item())
            return [audio_token_id] * count

        return [PromptReplacement(modality="audio", target=[audio_token_id], replacement=replacement)]

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        mm_res: MultiModalProcessingResult,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        num_audio = mm_items.get_all_counts().get("audio", 0)
        num_placeholders = mm_res.prompt_ids.count(self.info.get_hf_config().audio_token_id)
        if num_audio != 1 or num_placeholders != num_audio:
            raise ValueError(
                "VibeVoice accepts at most one reference audio, matched by exactly one unexpanded placeholder."
            )
        return super()._maybe_apply_prompt_updates(mm_items, mm_res)
