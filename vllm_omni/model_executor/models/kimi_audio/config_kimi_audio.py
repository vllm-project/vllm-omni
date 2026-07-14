# Copyright 2025 vLLM-Omni Team
"""Configuration classes for Kimi Audio model."""

from dataclasses import dataclass

from transformers import PretrainedConfig


@dataclass
class KimiAudioConfig(PretrainedConfig):
    """Configuration for Kimi Audio model.

    This config covers the main LLM model with MIMO layers.
    Audio detokenizer and vocoder configs are loaded separately from their subfolders.
    """

    # Main model config (from config.json)
    model_type: str = "kimi_audio"
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    intermediate_size: int = 18944
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-06
    vocab_size: int = 168448

    # Kimi Audio specific
    kimia_mimo_layers: int = 6
    kimia_mimo_transformer_from_layer_index: int = 21
    kimia_text_output_vocab: int = 152064
    kimia_audio_output_vocab: int = 16896
    kimia_token_offset: int = 152064

    # Whisper encoder config
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 128

    # Audio detokenizer config (from audio_detokenizer/config.yaml)
    dit_hidden_size: int = 2304
    dit_depth: int = 16
    dit_num_heads: int = 18
    dit_semantic_vocab_size: int = 16384
    dit_input_size: int = 80  # mel bins
    dit_ode_steps: int = 150
    dit_cfg_scale: float = 4.0

    # Vocoder config (from vocoder/config.json)
    vocoder_sampling_rate: int = 24000
    vocoder_hop_size: int = 480
    vocoder_num_mels: int = 80
    vocoder_upsample_rates: list = None

    def __post_init__(self, **kwargs):
        if self.vocoder_upsample_rates is None:
            self.vocoder_upsample_rates = [5, 2, 2, 2, 2, 3, 2]
        # Ignore extra kwargs from transformers

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load config from pretrained model path."""
        import json
        import os

        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        # Load audio detokenizer config
        detokenizer_config_path = os.path.join(pretrained_model_name_or_path, "audio_detokenizer", "config.yaml")
        if os.path.exists(detokenizer_config_path):
            import yaml

            with open(detokenizer_config_path) as f:
                detokenizer_config = yaml.safe_load(f)

            # Extract DiT config
            dit_config = detokenizer_config.get("model", {}).get("dit", {})
            config_dict["dit_hidden_size"] = dit_config.get("hidden_size", 2304)
            config_dict["dit_depth"] = dit_config.get("depth", 16)
            config_dict["dit_num_heads"] = dit_config.get("num_heads", 18)
            config_dict["dit_semantic_vocab_size"] = dit_config.get("semantic_vocab_size", 16384)
            config_dict["dit_input_size"] = dit_config.get("input_size", 80)
            config_dict["dit_ode_steps"] = detokenizer_config.get("ode_steps", 150)
            config_dict["dit_cfg_scale"] = detokenizer_config.get("cfg_scale", 4.0)

        # Load vocoder config
        vocoder_config_path = os.path.join(pretrained_model_name_or_path, "vocoder", "config.json")
        if os.path.exists(vocoder_config_path):
            with open(vocoder_config_path) as f:
                vocoder_config = json.load(f)

            config_dict["vocoder_sampling_rate"] = vocoder_config.get("sampling_rate", 24000)
            config_dict["vocoder_hop_size"] = vocoder_config.get("hop_size", 480)
            config_dict["vocoder_num_mels"] = vocoder_config.get("num_mels", 80)
            config_dict["vocoder_upsample_rates"] = vocoder_config.get("upsample_rates", [5, 2, 2, 2, 2, 3, 2])

        return cls(**config_dict)
