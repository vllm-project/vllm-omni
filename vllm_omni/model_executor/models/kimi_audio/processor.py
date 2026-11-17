# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cached audio preprocessing for Kimi's prepared, aligned token prompts."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from transformers import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser, ProcessorBatchItems
from vllm.multimodal.processing import BaseProcessingInfo, ProcessorInputs, PromptReplacement
from vllm.multimodal.processing.processor import MultiModalProcessingResult, PlaceholderFeaturesInfo

from vllm_omni.inputs.mm_processor import OmniDummyInputsBuilder, OmniMultiModalProcessor

from .audio_processing import SAMPLE_RATE, SAMPLES_PER_TOKEN, prepare_whisper_inputs


class KimiAudioDataParser(MultiModalDataParser):
    def _parse_audio_data(self, data):
        # Hash the encoding mode together with the waveform, before feature
        # extraction, to distinguish input audio from GLM-only history.
        if not isinstance(data, list) or not all(
            isinstance(item, dict)
            and set(item) == {"waveform", "use_whisper"}
            and isinstance(item["use_whisper"], bool)
            for item in data
        ):
            raise ValueError("Kimi-Audio requires audio items from prepare_kimi_audio_inputs")
        return ProcessorBatchItems(data, "audio") if data else None


class KimiAudioProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # vLLM queries input support while constructing its base ModelConfig,
        # before Omni attaches model_stage. Report model-level support then;
        # the stage-specific query below still excludes the acoustic decoder.
        if getattr(self.ctx.model_config, "model_stage", None) == "kimi_audio_decoder":
            return {}
        return {"audio": None}

    def get_data_parser(self) -> KimiAudioDataParser:
        return KimiAudioDataParser()

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        # Long recordings are chunked, not truncated at 30 seconds. A single
        # audio item may therefore occupy the entire available context.
        return {"audio": seq_len} if "audio" in self.supported_mm_limits else {}


class KimiAudioDummyInputsBuilder(OmniDummyInputsBuilder[KimiAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        count = mm_counts.get("audio", 0)
        if not count:
            return {}
        waveform = np.zeros(seq_len * SAMPLES_PER_TOKEN, dtype=np.float32)
        item = {
            "waveform": torch.from_numpy(waveform),
            "use_whisper": bool(self.info.get_hf_config().use_whisper_feature),
        }
        return {"audio": [item] * count}

    def get_dummy_processor_inputs(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, Any] | None = None
    ) -> ProcessorInputs:
        data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        count = mm_counts.get("audio", 0)
        offset = self.info.get_hf_config().kimia_token_offset
        # These tokens profile the encoder only; no request generation state
        # or artificial conversation is needed by the runner's dummy forward.
        return ProcessorInputs(
            prompt=[offset] * (seq_len * count) if count else [0],
            mm_data_items=self.info.parse_mm_data(data, validate=False),
        )


class KimiAudioMultiModalProcessor(OmniMultiModalProcessor[KimiAudioProcessingInfo]):
    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        mm_res: MultiModalProcessingResult,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        # prepare_kimi_audio_inputs already reserved each full, aligned audio
        # span. Locate the spans in item order; rewriting would desynchronize
        # the scheduled audio stream from the separate text stream.
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_res.kwargs, mm_item_counts)
        self._validate_mm_updates(mm_res.prompt_updates, mm_item_counts)
        mm_placeholders = self._find_mm_placeholders(mm_res.prompt_ids, mm_res.prompt_updates)
        self._validate_mm_placeholders(mm_placeholders, mm_item_counts)
        return mm_res.prompt_ids, mm_placeholders

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # With processor caching enabled, audios contains only cache misses.
        # Each item is self-contained; its index need not match the conversation.
        # The prepared prompt is already tokenized, so only audio is processed.
        if mm_kwargs:
            raise ValueError("Kimi-Audio does not support multimodal processor overrides")
        data: dict[str, Any] = {}
        audios = cast(list[dict[str, Any]], mm_data.get("audios", []))
        if audios:
            data.update(
                kimi_waveform=[item["waveform"] for item in audios],
                kimi_whisper_features=[],
                kimi_whisper_lengths=[],
            )
            extractor = None
            for item in audios:
                if item["use_whisper"]:
                    if extractor is None:
                        from vllm.transformers_utils.processor import cached_feature_extractor_from_config

                        extractor = cached_feature_extractor_from_config(
                            self.info.ctx.model_config, subfolder="whisper-large-v3"
                        )
                    whisper = prepare_whisper_inputs(item["waveform"].numpy(), extractor, sampling_rate=SAMPLE_RATE)
                    data["kimi_whisper_features"].append(whisper.input_features)
                    data["kimi_whisper_lengths"].append(torch.tensor(whisper.token_lengths, dtype=torch.long))
                else:
                    data["kimi_whisper_features"].append(torch.empty(0, 128, 3000))
                    data["kimi_whisper_lengths"].append(torch.empty(0, dtype=torch.long))
        return BatchFeature(data)

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        # encode_audio performs CPU GLM feature extraction and transfers mel
        # chunks itself. Avoid a redundant waveform GPU -> CPU round trip.
        return {
            f"kimi_{key}": MultiModalFieldConfig.batched("audio", keep_on_cpu=True)
            for key in ("waveform", "whisper_features", "whisper_lengths")
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs) -> Sequence[PromptReplacement]:
        waveforms = out_mm_kwargs.get_data().get("kimi_waveform", [])
        offset = self.info.get_hf_config().kimia_token_offset

        def audio_span(index: int) -> list[int]:
            count = (waveforms[index].numel() - 1) // SAMPLES_PER_TOKEN + 1
            return [offset] * count

        # The builder has ALREADY reserved these exact spans. The framework
        # discovers them in item order (see _maybe_apply_prompt_updates).
        return [PromptReplacement(modality="audio", target=[offset], replacement=audio_span)]
