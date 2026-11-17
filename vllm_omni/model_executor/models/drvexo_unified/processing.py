# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native image processing and the frozen 16kHz audio frontend."""

from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, BatchFeature, WhisperFeatureExtractor
from vllm.model_executor.models.qwen3_5 import Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_asr import _get_feat_extract_output_lengths
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import AudioEmbeddingItems, MultiModalDataParser
from vllm.multimodal.processing import BaseProcessingInfo, PromptReplacement

from vllm_omni.model_executor.models.drvexo_speech_8b.artifacts import resolve_artifact_path
from vllm_omni.model_executor.models.drvexo_speech_8b.processing import extract_audio_mels


class ImageAudioProcessor:
    def __init__(self, vision, feature_extractor):
        self.vision = vision
        self.feature_extractor = feature_extractor

    def __getattr__(self, key):
        return getattr(self.vision, key)

    def __call__(self, text=None, audios=None, audio=None, **kwargs):
        result = self.vision(text=text, **kwargs)
        # vLLM owns placeholder expansion, including cached-image requests.
        # This collapses the single image's patch run. The advertised limit of
        # one image is required: adjacent distinct images cannot share a run.
        if "input_ids" in result:
            image_id = self.vision.image_token_id
            ids = result["input_ids"]
            rows = ids.tolist() if isinstance(ids, torch.Tensor) else ids
            result["input_ids"] = [
                [token for i, token in enumerate(row) if token != image_id or i == 0 or row[i - 1] != image_id]
                for row in rows
            ]
        waveforms = audios if audios is not None else audio
        if waveforms is not None:
            if isinstance(waveforms, np.ndarray):
                waveforms = [waveforms]
            if len(waveforms):
                result["audio_mels"], result["audio_mel_lengths"] = extract_audio_mels(
                    self.feature_extractor, waveforms
                )
        return BatchFeature(result)


class UnifiedProcessingInfo(Qwen3_5ProcessingInfo):
    @cached_property
    def processor(self):
        root = Path(self.ctx.model_config.model)
        vision = AutoProcessor.from_pretrained(root / "assets/qwen35", local_files_only=True)
        vision.image_processor.size = {"longest_edge": 200704, "shortest_edge": 4096}
        audio = WhisperFeatureExtractor.from_pretrained(
            resolve_artifact_path(root, self.get_hf_config().audio_encoder_path), local_files_only=True
        )
        return ImageAudioProcessor(vision, audio)

    def get_hf_processor(self, **kwargs):
        return self.processor

    def get_supported_mm_limits(self):
        return {"image": 1, "audio": 1}

    def get_supported_mm_processor_kwargs(self):
        # Qwen3-VL adds video kwargs unconditionally. This stage advertises
        # images/audio only, so resolve precisely those supported HF scopes.
        return BaseProcessingInfo.get_supported_mm_processor_kwargs(self)

    def get_data_parser(self):
        return MultiModalDataParser(target_sr=16000, expected_hidden_size=2048)

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        result = super().get_mm_max_tokens_per_item(seq_len, {**mm_counts, "video": 0})
        return {"image": result["image"], "audio": 390}


class UnifiedDummyInputs(Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return "<|vision_start|><|image_pad|><|vision_end|>" * mm_counts.get(
            "image", 0
        ) + "<|audio_start|><|audio_pad|><|audio_end|>" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        result = super().get_dummy_mm_data(seq_len, {**mm_counts, "video": 0}, mm_options)
        if mm_counts.get("audio", 0):
            result["audio"] = [np.zeros(480000, dtype=np.float32)]
        return result


class UnifiedMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        return self.info.get_hf_processor()(text=prompt, **mm_data, **{**tok_kwargs, **mm_kwargs})

    def _hf_processor_applies_updates(self, *args, **kwargs):
        return False

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        result = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        result.update(
            {
                "audio_embeds": MultiModalFieldConfig.batched("audio"),
                "audio_mels": MultiModalFieldConfig.flat_from_sizes(
                    "audio", hf_inputs.get("audio_mel_lengths", torch.empty(0)), dim=1
                ),
                "audio_mel_lengths": MultiModalFieldConfig.batched("audio"),
            }
        )
        return result

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        result = [
            item
            for item in super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
            if item.modality == "image"
        ]
        if "audio" not in mm_items:
            return result
        items = mm_items["audio"]
        lengths = out_mm_kwargs.get_data().get("audio_mel_lengths", torch.empty(0, dtype=torch.long))
        output_lengths = _get_feat_extract_output_lengths(lengths)

        def replacement(index):
            count = int(
                items.get_feature_size(index) if isinstance(items, AudioEmbeddingItems) else output_lengths[index]
            )
            if not 0 < count <= 390:
                raise ValueError("Audio must contain 1–390 encoded frames")
            return [self.info.get_hf_config().audio_pad_token_id] * count

        result.append(
            PromptReplacement(
                modality="audio", target=[self.info.get_hf_config().audio_pad_token_id], replacement=replacement
            )
        )
        return result
