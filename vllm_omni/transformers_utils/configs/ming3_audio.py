# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Hugging Face configs for inclusionAI/Ming-3.0-audio-flash."""

from __future__ import annotations

from typing import Any, ClassVar

from transformers import AutoConfig, PretrainedConfig


class BailingMoeV3TextConfig(PretrainedConfig):
    """Native config adapter for the Bailing MoE v3 text backbone.

    The released checkpoint deliberately stores an empty ``model_type`` in
    ``text_config``. Normalize it to vLLM's canonical Bailing hybrid type so
    the shared architecture converter recognizes MLA and sizes its 576-wide
    latent KV cache correctly.
    """

    model_type = "bailing_hybrid"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)


class BailingMoeV3AudioEncoderConfig(PretrainedConfig):
    model_type = "bailing_moe_v3_audio_encoder"

    def __init__(
        self,
        n_mels: int = 128,
        n_audio_ctx: int = 3000,
        n_audio_state: int = 1280,
        n_audio_head: int = 20,
        n_audio_layer: int = 32,
        downsampler_kernel_size_list: list[int] | None = None,
        downsampler_stride_list: list[int] | None = None,
        downsampler_padding_list: list[int] | None = None,
        whisper_chunk_size: int = 3000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.downsampler_kernel_size_list = downsampler_kernel_size_list or [3]
        self.downsampler_stride_list = downsampler_stride_list or [2]
        if len(self.downsampler_kernel_size_list) != len(self.downsampler_stride_list):
            raise ValueError("Audio downsampler kernel and stride lists must have equal length")
        self.downsampler_padding_list = downsampler_padding_list or [
            kernel // 2 for kernel in self.downsampler_kernel_size_list
        ]
        if len(self.downsampler_padding_list) != len(self.downsampler_kernel_size_list):
            raise ValueError("Audio downsampler padding and kernel lists must have equal length")
        self.whisper_chunk_size = whisper_chunk_size


class BailingMoeV3AudioConfig(PretrainedConfig):
    model_type = "bailing_moe_v3_audio"
    is_composition = True
    sub_configs: ClassVar = {
        "text_config": BailingMoeV3TextConfig,
        "audio_config": BailingMoeV3AudioEncoderConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_config: dict[str, Any] | BailingMoeV3AudioEncoderConfig | None = None,
        audio_token_id: int = 157168,
        norm_query_embeds: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = BailingMoeV3TextConfig(**text_config)
        self.text_config = text_config or BailingMoeV3TextConfig()
        if isinstance(audio_config, dict):
            audio_config = BailingMoeV3AudioEncoderConfig(**audio_config)
        self.audio_config = audio_config or BailingMoeV3AudioEncoderConfig()
        self.audio_token_id = audio_token_id
        self.norm_query_embeds = norm_query_embeds
        # transformers>=5 validates token ids during base initialization and
        # calls get_text_config(), so nested configs must already be assigned.
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # noqa: ARG002
        return self.text_config


try:
    AutoConfig.register(BailingMoeV3AudioConfig.model_type, BailingMoeV3AudioConfig)
except ValueError:
    pass
