# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Full hybrid language/vision configuration for merged DRVEXO 4B."""

from transformers import PretrainedConfig, Qwen3_5Config


class DRVEXOUnifiedConfig(Qwen3_5Config):
    model_type = "drvexo_speech_qwen35_4b"

    def __init__(self, **kwargs):
        self.combine_text_audio_response = kwargs.pop("combine_text_audio_response", True)
        self.enable_audio_output = kwargs.pop("enable_audio_output", True)
        self.audio_encoder_path = kwargs.pop("audio_encoder_path", "audio_encoder")
        self.audio_encoder_config = kwargs.pop("audio_encoder_config", None)
        self.audio_feature_dim = kwargs.pop("audio_feature_dim", 2048)
        self.audio_pad_token_id = kwargs.pop("audio_pad_token_id", 248076)
        self.tts_model_path = kwargs.pop("tts_model_path", "talker")
        self.input_projector_type = kwargs.pop("input_projector_type", "temporal_residual_v1")
        self.temporal_projector_width = kwargs.pop("temporal_projector_width", 256)
        for key in ("talker_config", "code2wav_config"):
            value = kwargs.pop(key, None)
            setattr(self, key, PretrainedConfig(**value) if isinstance(value, dict) else value)
        super().__init__(**kwargs)
        self.hidden_size = self.text_config.hidden_size
