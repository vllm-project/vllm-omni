# SPDX-License-Identifier: Apache-2.0
"""Configs for Fun-Audio-Chat-8B. Mirrors the reference repo one-to-one so
`AutoConfig.from_pretrained(<ckpt>)` round-trips every field.

Reference: github.com/FunAudioLLM/Fun-Audio-Chat
  funaudiochat/configuration_funaudiochat.py
Checkpoint: FunAudioLLM/Fun-Audio-Chat-8B (config.json)
"""
from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

__all__ = [
    "FunAudioChatAudioEncoderConfig",
    "FunAudioChatConfig",
    # legacy alias kept so existing imports continue to work while we migrate
    "FunAudioChatAudioConfig",
]


class FunAudioChatAudioEncoderConfig(PretrainedConfig):
    """Audio-encoder sub-config (`funaudiochat_audio_encoder`).

    Covers both the continuous encoder (Whisper-style, custom attention with
    `cu_seqlens`) and the discrete encoder (`audio_tower`, embed + group pool).
    Also carries the CRQ decoder's sub-transformer config
    (`crq_transformer_config`) used by `FunAudioChatDecoder`.

    Fields mirror the reference class `FunAudioChatAudioEncoderConfig`.
    """

    model_type = "funaudiochat_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 4096,
        bos_token_id: int | None = None,
        codebook_size: int | None = None,
        continuous_features_mode: str = "replace",
        crq_transformer_config: dict | None = None,
        eos_token_id: int | None = None,
        group_size: int = 5,
        enable_audio_invert_tower: bool = True,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim

        self.bos_token_id = bos_token_id
        self.codebook_size = codebook_size
        self.continuous_features_mode = continuous_features_mode
        self.crq_transformer_config = crq_transformer_config
        self.eos_token_id = eos_token_id
        self.group_size = group_size
        self.enable_audio_invert_tower = enable_audio_invert_tower
        self.pad_token_id = pad_token_id


# Legacy alias — older adapter code imports `FunAudioChatAudioConfig`.
FunAudioChatAudioConfig = FunAudioChatAudioEncoderConfig


class FunAudioChatConfig(PretrainedConfig):
    """Top-level config (`funaudiochat`).

    Mirrors the reference class `FunAudioChatConfig` exactly so
    AutoConfig.from_pretrained round-trips the checkpoint config.json.
    """

    model_type = "funaudiochat"
    attribute_map = {"audio_token_id": "audio_token_index"}
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    def __init__(
        self,
        audio_config: "dict | FunAudioChatAudioEncoderConfig | None" = None,
        text_config: "dict | PretrainedConfig | None" = None,
        audio_token_index: int = 151669,
        ignore_index: int = -100,
        hidden_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.audio_token_index = audio_token_index
        self.ignore_index = ignore_index

        if isinstance(audio_config, dict):
            audio_config.setdefault("model_type", "funaudiochat_audio_encoder")
            audio_config = FunAudioChatAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = FunAudioChatAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config.setdefault("model_type", "qwen3")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen3"]()
        self.text_config = text_config

        self.hidden_size = hidden_size if hidden_size is not None else self.text_config.hidden_size

        super().__init__(**kwargs)

    def get_text_config(self, *_, **__):
        # transformers >=4.52 calls this with decoder=True; we ignore the hint.
        return self.text_config


# Register with transformers AutoConfig so vllm's get_config() path works.
AutoConfig.register("funaudiochat_audio_encoder", FunAudioChatAudioEncoderConfig)
AutoConfig.register("funaudiochat", FunAudioChatConfig)
