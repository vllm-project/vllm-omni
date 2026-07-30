# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm_omni.experimental.fullduplex.base.audio_utils import convert_token_to_id


class Qwen3OmniDuplexPolicy:
    SAMPLE_RATE_HZ = 16000
    CHUNK_SAMPLES = 16000
    # Whisper-based audio encoding: ~25 feature frames per second at 16kHz.
    SAMPLES_PER_AUDIO_TOKEN = 640

    IM_START_TOKEN_ID = 151644
    IM_END_TOKEN_ID = 151645

    AUDIO_START_TOKEN_ID = 151669
    AUDIO_END_TOKEN_ID = 151670
    AUDIO_PAD_TOKEN_ID = 151675

    TTS_BOS_TOKEN_ID = 151672
    TTS_EOS_TOKEN_ID = 151673
    TTS_PAD_TOKEN_ID = 151671

    TALKER_CODEC_PAD_TOKEN_ID = 4196
    TALKER_CODEC_BOS_TOKEN_ID = 4197
    TALKER_CODEC_EOS_TOKEN_ID = 4198

    SPECIAL_TOKEN_FIELDS: dict[str, str] = {
        "im_end_token_id": "<|im_end|>",
        "tts_bos_token_id": "<tts_text_bos>",
        "tts_eos_token_id": "<tts_text_eod>",
        "audio_start_token_id": "<|audio_start|>",
        "audio_end_token_id": "<|audio_end|>",
    }

    @classmethod
    def audio_token_count(cls, sample_count: int) -> int:
        return max(0, int(sample_count) // cls.SAMPLES_PER_AUDIO_TOKEN)

    @staticmethod
    def session_context_texts(instructions: object) -> tuple[str, str]:
        system_prompt = (
            instructions if isinstance(instructions, str) and instructions else "You are a helpful assistant."
        )
        prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        return prefix, suffix

    @classmethod
    def thinker_stop_token_ids(cls, tokenizer: Any) -> list[int]:
        out: list[int] = []
        im_end_id = convert_token_to_id(tokenizer, "<|im_end|>")
        if im_end_id is not None:
            out.append(im_end_id)
        return out

    @classmethod
    def scheduler_token_id(cls, tokenizer: Any) -> int | None:
        pad_id = convert_token_to_id(tokenizer, "<|audio_pad|>")
        if pad_id is not None:
            return pad_id
        eos_id = getattr(tokenizer, "eos_token_id", None)
        try:
            return int(eos_id)
        except (TypeError, ValueError):
            return None
