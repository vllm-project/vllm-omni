# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)


def ensure_funaudiochat_importable() -> ModuleType:
    try:
        import funaudiochat  # type: ignore

        return funaudiochat
    except ImportError:
        pass

    env_home = os.environ.get("FUN_AUDIO_CHAT_HOME")
    extra_candidates = [Path(env_home).expanduser()] if env_home else []

    for candidate in extra_candidates:
        if candidate and candidate.exists():
            sys.path.insert(0, str(candidate))
            try:
                import funaudiochat  # type: ignore

                return funaudiochat
            except ImportError:
                continue

    raise ImportError(
        "funaudiochat package is required. Install Fun-Audio-Chat into the active "
        "environment or set FUN_AUDIO_CHAT_HOME to the repo checkout."
    )


def resolve_funaudiochat_root() -> Path:
    pkg = ensure_funaudiochat_importable()
    pkg_path = Path(pkg.__file__).resolve()
    root = pkg_path.parent.parent
    if not root.exists():
        raise FileNotFoundError(f"Resolved Fun-Audio-Chat root does not exist: {root}")
    return root


class FunAudioChatProcessingInfo(BaseProcessingInfo):
    token_fps: int = 25

    @cached_property
    def feature_extractor(self) -> WhisperFeatureExtractor:
        return WhisperFeatureExtractor.from_pretrained(self.model_id)

    @cached_property
    def speech_tokenizer(self) -> PreTrainedTokenizerFast:
        return PreTrainedTokenizerFast.from_pretrained(self.model_id, subfolder="speech_tokenizer")

    def get_feature_extractor(self) -> WhisperFeatureExtractor:
        return self.feature_extractor

    def get_speech_tokenizer(self) -> PreTrainedTokenizerFast:
        return self.speech_tokenizer

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=int(self.feature_extractor.sampling_rate),
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int] | None:
        del seq_len, mm_counts
        cfg = self.get_hf_config()
        audio_cfg = getattr(cfg, "audio_config", None)
        max_audio_tokens = int(getattr(audio_cfg, "max_source_positions", 1500))
        return {"audio": max_audio_tokens}

    def get_audio_group_size(self) -> int:
        cfg = self.get_hf_config()
        audio_cfg = getattr(cfg, "audio_config", None)
        return int(getattr(audio_cfg, "group_size", 5))


class FunAudioChatDummyInputsBuilder(BaseDummyInputsBuilder[FunAudioChatProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<|audio_bos|><|AUDIO|><|audio_eos|>" * int(num_audios)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        del seq_len
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = int(feature_extractor.sampling_rate)
        cfg = self.info.get_hf_config()
        audio_cfg = getattr(cfg, "audio_config", None)
        max_audio_tokens = int(getattr(audio_cfg, "max_source_positions", 1500))
        group_size = self.info.get_audio_group_size()
        token_fps = int(getattr(self.info, "token_fps", 25))
        target_num_frames = max(1, max_audio_tokens) * max(1, group_size)
        audio_len = max(1, (target_num_frames * sampling_rate + token_fps - 1) // token_fps)
        num_audios = int(mm_counts.get("audio", 0))
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class FunAudioChatMultiModalProcessor(BaseMultiModalProcessor[FunAudioChatProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        del mm_kwargs
        tokenizer = self.info.get_tokenizer()
        text_inputs = tokenizer(
            prompt,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
            **tok_kwargs,
        )

        audios = mm_data.get("audios", [])
        if not audios:
            return BatchFeature(
                {
                    "input_ids": text_inputs["input_ids"],
                    "attention_mask": text_inputs["attention_mask"],
                }
            )

        feature_extractor = self.info.get_feature_extractor()
        sr = int(feature_extractor.sampling_rate)
        min_samples = int(getattr(feature_extractor, "n_fft", 400) or 400)

        wavs: list[np.ndarray] = []
        speech_strs: list[str] = []

        speech_tokenizer = self.info.get_speech_tokenizer()
        pad_token = speech_tokenizer.pad_token or "<|audio_pad|>"
        for audio in audios:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audio_np = np.asarray(audio, dtype=np.float32)
            if min_samples > 0 and audio_np.shape[0] < min_samples:
                audio_np = np.pad(audio_np, (0, min_samples - audio_np.shape[0]), mode="constant")

            wavs.append(audio_np)
            num_frames = int((float(audio_np.shape[0]) / float(sr)) * float(self.info.token_fps))
            speech_strs.append(pad_token * max(1, int(num_frames)))

        audio_group_size = self.info.get_audio_group_size()
        speech_inputs = speech_tokenizer(
            speech_strs,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            pad_to_multiple_of=audio_group_size,
            return_tensors="pt",
        )
        wav_inputs = feature_extractor(
            wavs,
            sampling_rate=sr,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )

        return BatchFeature(
            {
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
                "speech_ids": speech_inputs["input_ids"],
                "speech_attention_mask": speech_inputs["attention_mask"],
                "input_features": wav_inputs["input_features"],
                "feature_attention_mask": wav_inputs["attention_mask"],
                "feature_exist_mask": torch.ones((len(wavs),), dtype=torch.bool),
            }
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        del prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_inputs, hf_processor_mm_kwargs
        return {
            "speech_ids": MultiModalFieldConfig.batched("audio"),
            "speech_attention_mask": MultiModalFieldConfig.batched("audio"),
            "input_features": MultiModalFieldConfig.batched("audio"),
            "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
            "feature_exist_mask": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del hf_processor_mm_kwargs
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        audio_token = "<|AUDIO|>"
        audio_token_id = vocab[audio_token]
        out_mm_data = out_mm_kwargs.get_data()
        speech_attention_mask = out_mm_data.get("speech_attention_mask")
        if speech_attention_mask is None:
            audio_output_lengths: list[int] = []
        else:
            assert isinstance(speech_attention_mask, torch.Tensor)
            speech_lengths = speech_attention_mask.sum(-1)
            group_size = self.info.get_audio_group_size()
            audio_output_lengths = ((speech_lengths + group_size - 1) // group_size).tolist()

        def get_replacement(item_idx: int):
            num_features = int(audio_output_lengths[item_idx]) if audio_output_lengths else 1
            if num_features <= 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)
                raise ValueError(f"The audio (len={audio_len}) is too short to be represented inside the model")
            audio_tokens = [audio_token_id] * num_features
            return PromptUpdateDetails.select_token_id(audio_tokens, embed_token_id=audio_token_id)

        return [PromptReplacement(modality="audio", target=audio_token, replacement=get_replacement)]


def register_funaudiochat_processor(model_cls: type[Any]) -> type[Any]:
    return MULTIMODAL_REGISTRY.register_processor(
        FunAudioChatMultiModalProcessor,
        info=FunAudioChatProcessingInfo,
        dummy_inputs=FunAudioChatDummyInputsBuilder,
    )(model_cls)
