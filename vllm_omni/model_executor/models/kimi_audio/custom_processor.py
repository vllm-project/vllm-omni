# Copyright 2025 vLLM-Omni Team
"""Thin custom multi-modal processor for Kimi Audio.

Overrides the BLANK-count formula so the number of placeholder tokens equals
the number of embeddings produced by the Whisper encoder + 4-frame reshape.

GLM-4 tokenizer is NOT used here — it's only needed for voice cloning
(tokenizing reference audio), which is a separate pipeline path.
"""

from collections.abc import Sequence

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.kimi_audio import (
    _KIMIAUDIO_FIELD_CONFIG,
)
from vllm.model_executor.models.kimi_audio import (
    KimiAudioMultiModalProcessor as BaseKimiAudioMultiModalProcessor,
)
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import PromptReplacement
from vllm.transformers_utils.processors.kimi_audio import KimiAudioProcessor

from vllm_omni.model_executor.models.kimi_audio.constants import KIMI_AUDIO_BLANK_TOKEN_ID

logger = init_logger(__name__)


class CustomKimiAudioMultiModalProcessor(BaseKimiAudioMultiModalProcessor):
    """Custom processor with BLANK count matched to the encoder/reshape output."""

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs,
    ):
        """Expose the real frame count as an un-padded per-audio field."""
        config = dict(_KIMIAUDIO_FIELD_CONFIG)
        config["audio_real_frame_counts"] = MultiModalFieldConfig.batched("audio")
        return config

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ) -> Sequence[PromptReplacement]:
        """Compute BLANK count from the *real* mel-frame count.

        The Kimi Audio encoder pipeline is:

            real mel frames (T)
            -> Whisper encoder (conv1 stride=1, conv2 stride=2) -> ceil(T/2)
            -> 4-frame reshape -> ceil(ceil(T/2)/4) = ceil(T/8)

        The processor pads all inputs to the same fixed length (typically
        3000 frames), so ``whisper_input_features.shape[-1]`` is the padded
        length, not the audio length.  Using the padded length creates far
        too many BLANK tokens and feeds silence into the model, destroying
        ASR quality.  We therefore use ``feature_attention_mask.sum(-1)``,
        which counts the real non-padded frames per batch item.
        """
        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")

        if feature_attention_mask is not None and hasattr(feature_attention_mask, "shape"):
            # Per-item real frame counts: [num_audio_items]
            real_frame_counts = feature_attention_mask.sum(dim=-1).tolist()
            audio_output_lengths = [max(1, (int(count) + 7) // 8) for count in real_frame_counts]
        else:
            # Fallback: padded feature length (one global value)
            whisper_features = out_mm_data.get("whisper_input_features")
            if whisper_features is not None and hasattr(whisper_features, "shape"):
                num_mel_frames = int(whisper_features.shape[-1])
            else:
                num_mel_frames = KimiAudioProcessor.AUDIO_SEQ_LEN
            audio_output_lengths = [max(1, (num_mel_frames + 7) // 8)]

        # Pass the real (un-padded) mel-frame counts through to the model so it
        # can truncate the padded Whisper features before the audio tower.  The
        # field config declares this as a batched per-audio scalar.
        audio_items = list(out_mm_kwargs.get("audio", []))
        if audio_items and feature_attention_mask is not None:
            counts_tensor = torch.tensor(
                [int(feature_attention_mask[b].sum().item()) for b in range(feature_attention_mask.shape[0])],
                dtype=torch.long,
            )
            count_elems = MultiModalFieldConfig.batched("audio").build_elems("audio_real_frame_counts", counts_tensor)
            for item, elem in zip(audio_items, count_elems):
                if item is not None:
                    item["audio_real_frame_counts"] = elem

        logger.debug(
            "[CUSTOM-PROCESSOR] audio_output_lengths=%s, target_token=%d, feature_attention_mask=%s",
            audio_output_lengths,
            KIMI_AUDIO_BLANK_TOKEN_ID,
            feature_attention_mask.shape if hasattr(feature_attention_mask, "shape") else None,
        )

        def get_replacement_kimiaudio(item_idx: int):
            num_features = (
                audio_output_lengths[item_idx] if item_idx < len(audio_output_lengths) else audio_output_lengths[-1]
            )
            return [KIMI_AUDIO_BLANK_TOKEN_ID] * num_features

        return [
            PromptReplacement(
                modality="audio",
                target=[KIMI_AUDIO_BLANK_TOKEN_ID],
                replacement=get_replacement_kimiaudio,
            ),
        ]
