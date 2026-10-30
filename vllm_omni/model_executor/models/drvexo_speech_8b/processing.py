# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Audio processing for DRVEXO's shared Qwen text backbone.

Raw audio uses the native Qwen3-ASR mel front end. Precomputed audio_embeds
are ASR encoder states (2048), not language embeddings (4096).
"""

from functools import cached_property

import numpy as np
import torch
from transformers import BatchFeature, ProcessorMixin, WhisperFeatureExtractor
from vllm.model_executor.models.qwen3_asr import _get_feat_extract_output_lengths
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import AudioEmbeddingItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)

from .artifacts import resolve_artifact_path


def extract_audio_mels(feature_extractor, audios):
    """Shared offline/serving frontend; encoded training features use this too."""
    for audio in audios:
        if not 160 <= len(audio) <= 480000 or not np.isfinite(audio).all():
            raise ValueError("DRVEXO audio must be finite and between 0.01 and 30 seconds")
    features = feature_extractor(
        audios,
        sampling_rate=16000,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    lengths = features.attention_mask.sum(-1)
    mels = torch.cat([feat[:, : int(length)] for feat, length in zip(features.input_features, lengths)], dim=1)
    return mels, lengths


class DRVEXOHFProcessor(ProcessorMixin):
    """Explicit text/audio processor, including vLLM's media-only calls."""

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, text=None, audios=None, audio=None, **kwargs):
        result = BatchFeature({})
        if text is not None:
            texts = [text] if isinstance(text, str) else text
            result["input_ids"] = [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]
        waveforms = audios if audios is not None else audio
        if waveforms is not None:
            if isinstance(waveforms, np.ndarray):
                waveforms = [waveforms]
            if len(waveforms):
                mels, lengths = extract_audio_mels(self.feature_extractor, waveforms)
                result["audio_mels"] = mels
                result["audio_mel_lengths"] = lengths
        return result


class DRVEXOProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs):
        return DRVEXOHFProcessor(self.feature_extractor, self.get_tokenizer())

    def get_supported_mm_limits(self):
        return {"audio": 1}

    def get_data_parser(self):
        return MultiModalDataParser(target_sr=16000, expected_hidden_size=2048)

    @cached_property
    def feature_extractor(self):
        return WhisperFeatureExtractor.from_pretrained(
            resolve_artifact_path(self.ctx.model_config.model, self.ctx.model_config.hf_config.audio_encoder_path),
            local_files_only=True,
        )

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        # This experimental deployment caps raw clips at thirty seconds.
        return {"audio": 390}


class DRVEXODummyInputsBuilder(BaseDummyInputsBuilder[DRVEXOProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        return "<|audio_start|><|audio_pad|><|audio_end|>" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        return {"audio": [np.zeros(480000, dtype=np.float32) for _ in range(mm_counts.get("audio", 0))]}


class DRVEXOMultiModalProcessor(BaseMultiModalProcessor[DRVEXOProcessingInfo]):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        # vLLM 0.28 combines text/media here; 0.29 calls get_hf_processor
        # for media separately. Both paths use the same frontend.
        return self.info.get_hf_processor()(text=prompt, audios=mm_data.get("audios", []))

    def _hf_processor_applies_updates(self, *args, **kwargs):
        return False

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {
            "audio_embeds": MultiModalFieldConfig.batched("audio"),
            "audio_mels": MultiModalFieldConfig.flat_from_sizes(
                "audio", hf_inputs.get("audio_mel_lengths", torch.empty(0)), dim=1
            ),
            "audio_mel_lengths": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        token_id = self.info.ctx.model_config.hf_config.audio_pad_token_id
        items = mm_items.get("audio")
        data = out_mm_kwargs.get_data()
        lengths = data.get("audio_mel_lengths", torch.empty(0, dtype=torch.long))
        output_lengths = _get_feat_extract_output_lengths(lengths).tolist()

        def replacement(index):
            count = items.get_feature_size(index) if isinstance(items, AudioEmbeddingItems) else output_lengths[index]
            if not 0 < count <= 390:
                raise ValueError("DRVEXO audio must contain 1–390 encoded frames")
            return [token_id] * count

        return [PromptReplacement(modality="audio", target=[token_id], replacement=replacement)]
