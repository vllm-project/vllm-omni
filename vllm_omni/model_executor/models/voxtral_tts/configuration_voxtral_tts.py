from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import MistralConfigParser, register_config_parser

logger = init_logger(__name__)


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


@register_config_parser("mistral")
class VoxtralTTSConfigParser(MistralConfigParser):
    """Config parser that extends the base Mistral parser with TTS support.

    This only support voxtral_tts for now.
    """

    def _remap_mistral_audio_args(self, config_dict: dict) -> dict:
        encoder_args = config_dict["multimodal"].pop("audio_model_args")
        audio_tokenizer_args = config_dict["multimodal"].pop("audio_tokenizer_args", None)
        audio_config = {}
        if encoder_args is not None:
            # Default n_decoding_steps if not provided
            acoustic_args = encoder_args.get("acoustic_transformer_args", {})
            if acoustic_args.get("n_decoding_steps") is None:
                logger.warning(
                    "n_decoding_steps not provided in acoustic_transformer_args, defaulting to 7. "
                    "Please add 'n_decoding_steps' to params.json under acoustic_transformer_args."
                )
                acoustic_args["n_decoding_steps"] = 7

            audio_config = {
                "sampling_rate": encoder_args["audio_encoding_args"]["sampling_rate"],
                "codec_args": audio_tokenizer_args,
                "audio_model_args": encoder_args,
                "speaker_id": audio_tokenizer_args.get("voice", {}),
            }
        return audio_config

    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, PretrainedConfig]:
        from vllm.transformers_utils.config import (
            _download_mistral_config_file,
        )

        config_dict = _download_mistral_config_file(model, revision)

        from vllm.transformers_utils.configs.mistral import (
            _remap_general_mistral_args,
            _remap_mistral_quantization_args,
        )

        # Extract audio config before building text config
        audio_config = {}
        if (config_dict.get("multimodal") or {}).get("audio_model_args"):
            audio_config = self._remap_mistral_audio_args(config_dict)

        # Build text_config from the top-level keys
        non_text_keys = {"multimodal"}
        text_config = {k: v for k, v in config_dict.items() if k not in non_text_keys}
        text_config = _remap_general_mistral_args(text_config)
        if text_config.get("quantization"):
            text_config = _remap_mistral_quantization_args(text_config)

        # The text sub-model is a plain MistralForCausalLM
        text_config.setdefault("architectures", ["MistralForCausalLM"])

        config = VoxtralTTSConfig(
            text_config=PretrainedConfig.from_dict(text_config),
            audio_config=audio_config,
            architectures=config_dict.get("architectures", ["VoxtralTTSForConditionalGeneration"]),
        )

        return config_dict, config
