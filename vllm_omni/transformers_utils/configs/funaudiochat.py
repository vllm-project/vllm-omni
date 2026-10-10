# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transformers configuration registration for Fun-Audio-Chat."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


class FunAudioChatAudioEncoderConfig(PretrainedConfig):
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
        output_dim: int = 3584,
        bos_token_id: int | None = None,
        codebook_size: int | None = None,
        continuous_features_mode: str = "replace",
        crq_transformer_config: dict[str, Any] | None = None,
        eos_token_id: int | None = None,
        group_size: int = 5,
        enable_audio_invert_tower: bool = True,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.encoder_layers = encoder_layers
        self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.d_model = d_model
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.scale_embedding = scale_embedding
        self.initializer_range = initializer_range
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


class FunAudioChatConfig(PretrainedConfig):
    model_type = "funaudiochat"
    attribute_map = {"audio_token_id": "audio_token_index"}

    def __init__(
        self,
        audio_config: PretrainedConfig | dict[str, Any] | None = None,
        text_config: PretrainedConfig | dict[str, Any] | None = None,
        audio_token_index: int = 151646,
        ignore_index: int = -100,
        hidden_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        self.audio_token_index = audio_token_index
        self.ignore_index = ignore_index

        if isinstance(audio_config, dict):
            audio_config = {
                **audio_config,
                "model_type": audio_config.get("model_type", FunAudioChatAudioEncoderConfig.model_type),
            }
            if audio_config["model_type"] == FunAudioChatAudioEncoderConfig.model_type:
                audio_config.pop("model_type")
                audio_config = FunAudioChatAudioEncoderConfig(**audio_config)
            else:
                model_type = audio_config.pop("model_type")
                audio_config = CONFIG_MAPPING[model_type](**audio_config)
        elif audio_config is None:
            audio_config = FunAudioChatAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            model_type = text_config.pop("model_type", "qwen3")
            text_config = CONFIG_MAPPING[model_type](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen3"]()
        self.text_config = text_config

        self.hidden_size = int(text_config.hidden_size) if hidden_size is None else int(hidden_size)
        super().__init__(**kwargs)


AutoConfig.register(
    FunAudioChatAudioEncoderConfig.model_type,
    FunAudioChatAudioEncoderConfig,
    exist_ok=True,
)
AutoConfig.register(FunAudioChatConfig.model_type, FunAudioChatConfig, exist_ok=True)

__all__ = ["FunAudioChatAudioEncoderConfig", "FunAudioChatConfig"]
