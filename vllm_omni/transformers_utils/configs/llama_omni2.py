# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Local Transformers config for ICTNLP LLaMA-Omni 2 checkpoints."""

from __future__ import annotations

import ast
from typing import Any

from transformers import AutoConfig, PretrainedConfig, Qwen2Config

__all__ = ["LlamaOmni2Config"]


def _build_qwen2_config(
    raw: dict[str, Any] | PretrainedConfig | None,
    *,
    fallback: dict[str, Any] | None = None,
) -> Qwen2Config:
    if isinstance(raw, Qwen2Config):
        return raw
    if isinstance(raw, PretrainedConfig):
        return Qwen2Config(**raw.to_dict())
    values = dict(fallback or {})
    values.update(dict(raw or {}))
    values.pop("model_type", None)
    return Qwen2Config(**values)


def _parse_stream_params(value: str | tuple[int, int] | list[int]) -> tuple[int, int]:
    parsed: Any = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"stream_params must be a pair of positive integers, got {value!r}") from exc

    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"stream_params must be a pair of positive integers, got {value!r}")

    text_tokens, unit_tokens = parsed
    if (
        isinstance(text_tokens, bool)
        or isinstance(unit_tokens, bool)
        or not isinstance(text_tokens, int)
        or not isinstance(unit_tokens, int)
        or text_tokens <= 0
        or unit_tokens <= 0
    ):
        raise ValueError(f"stream_params must be a pair of positive integers, got {value!r}")
    return text_tokens, unit_tokens


class LlamaOmni2Config(Qwen2Config):
    """Typed root config for the LLaMA-Omni 2 Thinker/Talker pipeline."""

    model_type = "omni2_speech2s_qwen2"
    is_composition = True

    def __init__(
        self,
        speech_generator: dict[str, Any] | PretrainedConfig | None = None,
        thinker_config: dict[str, Any] | PretrainedConfig | None = None,
        talker_config: dict[str, Any] | PretrainedConfig | None = None,
        speech_encoder: str | None = None,
        speech_encoder_type: str = "whisper",
        speech_encoder_ds_rate: int = 5,
        speech_encoder_hidden_size: int = 1280,
        speech_projector_type: str = "linear",
        stream_params: str | tuple[int, int] | list[int] = "(3,10)",
        unit_vocab_size: int = 6561,
        tokenizer_model_max_length: int = 4096,
        tokenizer_padding_side: str = "right",
        **kwargs: Any,
    ) -> None:
        root_qwen_values = dict(kwargs)
        self.thinker_config = _build_qwen2_config(thinker_config, fallback=root_qwen_values)
        self.talker_config = _build_qwen2_config(
            talker_config if talker_config is not None else speech_generator,
        )
        self.stream_text_tokens, self.stream_unit_tokens = _parse_stream_params(stream_params)

        super().__init__(**kwargs)

        self.speech_generator = self.talker_config.to_dict()

        self.speech_encoder = speech_encoder
        self.speech_encoder_type = speech_encoder_type
        self.speech_encoder_ds_rate = int(speech_encoder_ds_rate)
        self.speech_encoder_hidden_size = int(speech_encoder_hidden_size)
        self.speech_projector_type = speech_projector_type
        self.unit_vocab_size = int(unit_vocab_size)
        self.tokenizer_model_max_length = int(tokenizer_model_max_length)
        self.tokenizer_padding_side = tokenizer_padding_side

        self.stream_params = f"({self.stream_text_tokens},{self.stream_unit_tokens})"

    def get_text_config(self, decoder: bool = False) -> Qwen2Config:
        del decoder
        return self.thinker_config


AutoConfig.register(LlamaOmni2Config.model_type, LlamaOmni2Config)
