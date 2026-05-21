# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# MiniMind-O multimodal processor.

from collections.abc import Mapping
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerProcessingInfo,
    Qwen2_5OmniThinkerMultiModalProcessor as _Qwen2_5OmniThinkerMultiModalProcessorBase,
)
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)

logger = init_logger(__name__)


class MiniMindOThinkerProcessingInfo(Qwen2_5OmniThinkerProcessingInfo):
    """MiniMind-O specific processing info."""
    
    def get_hf_config(self):
        """Get MiniMind-O config."""
        return self.model_config.hf_config


class MiniMindOThinkerMultiModalProcessor(_Qwen2_5OmniThinkerMultiModalProcessorBase):
    """MiniMind-O specific multimodal processor.
    
    Inherits from Qwen2.5-Omni base processor with MiniMind-O specific
    token handling for audio and vision.
    """

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """Apply prompt updates for MiniMind-O specific tokens."""
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)
        self._validate_mm_updates(mm_prompt_updates, mm_item_counts)

        # MiniMind-O uses simpler audio/vision token handling
        # No interleaved audio-in-video like Qwen2.5-Omni
        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )
        else:
            prompt_ids, mm_placeholders = self._apply_prompt_updates(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )

        return prompt_ids, mm_placeholders


class MiniMindOThinkerDummyInputsBuilder(Qwen2_5OmniThinkerDummyInputsBuilder):
    """MiniMind-O specific dummy inputs builder."""
    
    def get_dummy_audio_inputs(self, seq_len: int, num_audio: int):
        """Get dummy audio inputs for MiniMind-O."""
        # MiniMind-O uses audio_pad_token (2049)
        return super().get_dummy_audio_inputs(seq_len, num_audio)
    
    def get_dummy_image_inputs(self, seq_len: int, num_images: int):
        """Get dummy image inputs for MiniMind-O."""
        # MiniMind-O uses image_pad_token (12)
        return super().get_dummy_image_inputs(seq_len, num_images)
