from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


class VoxtralTTSConfig(PretrainedConfig):
    """HuggingFace-style config for Voxtral TTS models."""

    model_type = "voxtral_tts"

    def __init__(
        self,
        text_config: PretrainedConfig | dict | None = None,
        audio_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        elif isinstance(text_config, dict):
            self.text_config = PretrainedConfig.from_dict(text_config)
        else:
            self.text_config = PretrainedConfig()

        self.audio_config = audio_config or {}

    def get_text_config(self, **kwargs: Any) -> PretrainedConfig:
        return self.text_config


AutoConfig.register("voxtral_tts", VoxtralTTSConfig)

__all__ = ["VoxtralTTSConfig"]
