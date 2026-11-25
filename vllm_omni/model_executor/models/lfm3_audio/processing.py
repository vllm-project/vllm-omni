# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio preprocessing: the LFM3-Omni processor for audio-only checkpoints."""

from collections.abc import Mapping

from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails

from vllm_omni.model_executor.models.lfm3_omni.processing import (
    Lfm3OmniDummyInputsBuilder,
    Lfm3OmniMultiModalProcessor,
    Lfm3OmniProcessingInfo,
)


class Lfm3AudioProcessingInfo(Lfm3OmniProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class Lfm3AudioDummyInputsBuilder(Lfm3OmniDummyInputsBuilder):
    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        # 1280 samples at 16 kHz = one encoder token (hop 160, stride 8).
        return {
            "audio": self._get_dummy_audios(
                length=seq_len * 1280,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class Lfm3AudioMultiModalProcessor(Lfm3OmniMultiModalProcessor):
    """LFM3-Audio expands each clip to bare `<|audio|>` tokens, without markers."""

    def _postprocess_prompt(self, prompt):
        return prompt

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        if "audio" not in out_mm_kwargs:
            return []
        token_id = self.info.get_hf_config().audio_token_id
        extractor = self.info.get_hf_processor(**hf_processor_mm_kwargs).feature_extractor
        mask = out_mm_kwargs.get_data()["feature_attention_mask"]
        lengths = [extractor.num_encoder_tokens(int(n)) for n in mask.sum(-1)]

        def replacement(index):
            return PromptUpdateDetails.select_token_id([token_id] * lengths[index], embed_token_id=token_id)

        return [PromptReplacement(modality="audio", target=[token_id], replacement=replacement)]
