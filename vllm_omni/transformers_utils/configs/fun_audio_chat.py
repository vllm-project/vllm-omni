# SPDX-License-Identifier: Apache-2.0
from typing import Any

from transformers import AutoConfig, PretrainedConfig

__all__ = ["FunAudioChatAudioConfig", "FunAudioChatConfig"]


class FunAudioChatAudioConfig(PretrainedConfig):
    """Audio-encoder sub-config for Fun-Audio-Chat-8B (funaudiochat_audio_encoder)."""

    model_type = "funaudiochat_audio_encoder"

    def __init__(
        self,
        d_model: int = 1280,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        num_mel_bins: int = 128,
        output_dim: int = 4096,
        max_source_positions: int = 1500,
        group_size: int = 5,
        activation_function: str = "gelu",
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        scale_embedding: bool = False,
        n_window: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_mel_bins = num_mel_bins
        self.output_dim = output_dim
        self.max_source_positions = max_source_positions
        self.group_size = group_size
        self.activation_function = activation_function
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.scale_embedding = scale_embedding
        self.n_window = n_window


class FunAudioChatConfig(PretrainedConfig):
    """Top-level config for Fun-Audio-Chat-8B (funaudiochat model_type)."""

    model_type = "funaudiochat"
    sub_configs = {"audio_config": FunAudioChatAudioConfig}

    def __init__(
        self,
        audio_config: "dict | FunAudioChatAudioConfig | None" = None,
        text_config: "dict | None" = None,
        audio_token_index: int = 151669,
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(audio_config, dict):
            # Pass all fields; S2S needs crq_transformer_config, codebook_size, bos_token_id.
            # Unknown fields go to **kwargs and are stored by PretrainedConfig via setattr.
            audio_config = FunAudioChatAudioConfig(**audio_config)
        elif audio_config is None:
            audio_config = FunAudioChatAudioConfig()
        self.audio_config = audio_config

        # text_config: store as a PretrainedConfig so hf_text_config resolution works
        if isinstance(text_config, dict):
            # Import Qwen3 config at runtime to avoid circular imports
            try:
                from transformers import AutoConfig as _AC
                self.text_config = _AC.for_model("qwen3", **text_config)
            except Exception:
                self.text_config = PretrainedConfig(**text_config)
        else:
            self.text_config = text_config or PretrainedConfig()

        self.audio_token_index = audio_token_index
        self.ignore_index = ignore_index

    def get_text_config(self):
        return self.text_config


# Register with transformers AutoConfig so vllm's get_config() path works.
AutoConfig.register("funaudiochat", FunAudioChatConfig)
