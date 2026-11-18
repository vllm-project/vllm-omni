# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reuse upstream Qwen3-ASR inference and expose its detected language."""

import regex as re
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration as VllmQwen3ASRForConditionalGeneration,
)

from vllm_omni.model_executor.models.interfaces import TranscriptionLanguageMixin


class Qwen3ASRForConditionalGeneration(VllmQwen3ASRForConditionalGeneration, TranscriptionLanguageMixin):
    @classmethod
    def get_transcription_language(cls, raw_text: str) -> str | None:
        if "<asr_text>" not in raw_text:
            return None
        prefix, _ = raw_text.strip().rsplit("<asr_text>", 1)
        match = re.search(r"(?:^|\n)language\s+([^<\n]+)\s*$", prefix, flags=re.IGNORECASE)
        if match:
            language = match.group(1).strip()
            if language and language.lower() not in {"none", "unknown", "auto"}:
                return language
        return None
