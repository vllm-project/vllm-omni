# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for Fun-Audio-Chat model.

This provides a proper PretrainedConfig subclass that correctly parses
the nested configuration (text_config, audio_config) from the model's config.json.
"""

from transformers import PretrainedConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class FunAudioChatAudioConfig(PretrainedConfig):
    """Audio configuration for Fun-Audio-Chat."""

    model_type = "funaudiochat_audio"

    def __init__(
        self,
        d_model: int = 1280,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        encoder_layers: int = 32,
        num_mel_bins: int = 128,
        max_source_positions: int = 1500,
        feature_projection_dim: int = 1024,
        codebook_size: int = 6565,
        group_size: int = 5,
        bos_token_id: int = 6561,
        eos_token_id: int = 6562,
        padding_token_id: int = 6564,
        output_dim: int = 4096,
        crq_hidden_size: int = 1024,
        crq_num_heads: int = 16,
        crq_num_layers: int = 28,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.feature_projection_dim = feature_projection_dim
        self.codebook_size = codebook_size
        self.group_size = group_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.padding_token_id = padding_token_id
        self.output_dim = output_dim
        self.crq_hidden_size = crq_hidden_size
        self.crq_num_heads = crq_num_heads
        self.crq_num_layers = crq_num_layers


class FunAudioChatConfig(PretrainedConfig):
    """Configuration class for Fun-Audio-Chat model.

    This handles the nested text_config and audio_config properly,
    converting them from dicts to PretrainedConfig objects.
    """

    model_type = "funaudiochat"
    sub_configs = {"text_config": PretrainedConfig, "audio_config": FunAudioChatAudioConfig}

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        audio_config: dict | FunAudioChatAudioConfig | None = None,
        audio_token_index: int = 151669,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.audio_token_index = audio_token_index
        self.ignore_index = ignore_index

        # Parse text_config
        if text_config is None:
            # Default Qwen3 8B config
            self.text_config = PretrainedConfig(
                model_type="qwen3",
                hidden_size=4096,
                intermediate_size=12288,
                num_hidden_layers=36,
                num_attention_heads=32,
                num_key_value_heads=8,
                vocab_size=151936,
                max_position_embeddings=40960,
            )
        elif isinstance(text_config, dict):
            # Convert dict to PretrainedConfig
            text_model_type = text_config.get("model_type", "qwen3")
            self.text_config = PretrainedConfig(**text_config)
            self.text_config.model_type = text_model_type
        else:
            self.text_config = text_config

        # Parse audio_config
        if audio_config is None:
            self.audio_config = FunAudioChatAudioConfig()
        elif isinstance(audio_config, dict):
            self.audio_config = FunAudioChatAudioConfig(**audio_config)
        else:
            self.audio_config = audio_config

    def get_text_config(self) -> PretrainedConfig:
        """Return the text config for compatibility with vLLM."""
        return self.text_config

    @property
    def vocab_size(self) -> int:
        """Return vocab size from text config."""
        return getattr(self.text_config, "vocab_size", 151936)

    @property
    def hidden_size(self) -> int:
        """Return hidden size from text config."""
        return getattr(self.text_config, "hidden_size", 4096)

    @property
    def num_hidden_layers(self) -> int:
        """Return number of hidden layers from text config."""
        return getattr(self.text_config, "num_hidden_layers", 36)

    @property
    def num_attention_heads(self) -> int:
        """Return number of attention heads from text config."""
        return getattr(self.text_config, "num_attention_heads", 32)

    @property
    def num_key_value_heads(self) -> int:
        """Return number of key-value heads from text config."""
        return getattr(self.text_config, "num_key_value_heads", 8)

    @property
    def max_position_embeddings(self) -> int:
        """Return max position embeddings from text config."""
        return getattr(self.text_config, "max_position_embeddings", 40960)


# Register FunAudioChatConfig to Transformers
def register_funaudiochat_config():
    """Register FunAudioChatConfig to Transformers' CONFIG_MAPPING."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        try:
            _ = CONFIG_MAPPING["funaudiochat"]
        except KeyError:
            CONFIG_MAPPING.register("funaudiochat", FunAudioChatConfig, exist_ok=True)
            logger.debug("Registered FunAudioChatConfig to Transformers CONFIG_MAPPING")
    except Exception as e:
        logger.warning(f"Failed to register FunAudioChatConfig: {e}")


__all__ = ["FunAudioChatConfig", "FunAudioChatAudioConfig", "register_funaudiochat_config"]
