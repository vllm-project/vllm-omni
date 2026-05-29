"""KimiAudio multimodal processor: tokenize input audio with GLM-4-Voice
and route the resulting codec IDs through the audio-stream placeholder
expansion. Mirrors upstream Moonshot's dual representation — discrete
codec tokens on the audio stream, continuous Whisper features fused on
top via ``(audio + whisper) * sqrt(2)``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from transformers import BatchFeature
from vllm.model_executor.models.kimi_audio import (
    KimiAudioMultiModalProcessor as _UpstreamProcessor,
)
from vllm.model_executor.models.kimi_audio import (
    _get_feat_extract_output_lengths,
)
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import PromptReplacement
from vllm.transformers_utils.processors.kimi_audio import KimiAudioProcessor

from .glm4_voice_tokenizer import tokenize_audio

_GLM4_FIELD = "glm4_audio_token_ids"
_GLM4_LEN_FIELD = "glm4_audio_token_lens"
_BLANK = KimiAudioProcessor.KIMIA_TEXT_BLANK


class OmniKimiAudioMultiModalProcessor(_UpstreamProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        feature = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        audios = mm_data.get("audios") or mm_data.get("audio") or []
        if not audios:
            return feature

        # Normalize: real inputs come as ``(np.ndarray, sample_rate)``;
        # vLLM's profile_run dummy data sends bare np.ndarray at 16 kHz.
        token_lists = [tokenize_audio(*(a if isinstance(a, tuple) else (a, 16000))) for a in audios]
        feature[_GLM4_FIELD] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in token_lists],
            batch_first=True,
            padding_value=_BLANK,
        )
        feature[_GLM4_LEN_FIELD] = torch.tensor([len(ids) for ids in token_lists], dtype=torch.long)
        return feature

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {
            "whisper_input_features": MultiModalFieldConfig.batched("audio"),
            "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
            _GLM4_FIELD: MultiModalFieldConfig.batched("audio"),
            _GLM4_LEN_FIELD: MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ) -> Sequence[PromptReplacement]:
        out = out_mm_kwargs.get_data()
        feature_attention_mask = out.get("feature_attention_mask")
        if feature_attention_mask is None:
            return []  # text-only path; nothing to splice in
        whisper_lens = _get_feat_extract_output_lengths(feature_attention_mask.sum(-1)).tolist()
        glm4_tokens = out[_GLM4_FIELD]
        glm4_lens = out[_GLM4_LEN_FIELD]

        def get_replacement(item_idx: int) -> list[int]:
            # Whisper feature length is the source of truth for the
            # placeholder expansion count; pad/truncate the GLM-4-Voice
            # IDs to match so the (audio + whisper) fusion lines up.
            num_features = whisper_lens[item_idx]
            length = int(glm4_lens[item_idx].item())
            ids = glm4_tokens[item_idx, :length].tolist()
            if len(ids) < num_features:
                ids = ids + [_BLANK] * (num_features - len(ids))
            else:
                ids = ids[:num_features]
            return ids

        return [PromptReplacement(modality="audio", target=[_BLANK], replacement=get_replacement)]
