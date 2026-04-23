from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    _CONFIG_FORMAT_TO_CONFIG_PARSER,
    MistralConfigParser,
    _download_mistral_config_file,
)

from vllm_omni.transformers_utils.configs.voxtral_tts import VoxtralTTSConfig

logger = init_logger(__name__)

_VOXTRAL_TTS_ARCHS = frozenset({"VoxtralTTSForConditionalGeneration"})
_VOXTRAL_TTS_MODEL_TYPE = "voxtral_tts"


def _is_voxtral_tts_params(config_dict: dict) -> bool:
    """Return True if the Mistral params.json describes a Voxtral-TTS model"""
    if config_dict.get("model_type") == _VOXTRAL_TTS_MODEL_TYPE:
        return True
    architectures = set(config_dict.get("architectures") or [])
    return bool(architectures & _VOXTRAL_TTS_ARCHS)


def _remap_voxtral_tts_audio_args(config_dict: dict) -> dict:
    encoder_args = config_dict["multimodal"].pop("audio_model_args")
    audio_tokenizer_args = config_dict["multimodal"].pop("audio_tokenizer_args", None)
    if encoder_args is None:
        return {}

    acoustic_args = encoder_args.get("acoustic_transformer_args", {})
    if acoustic_args.get("n_decoding_steps") is None:
        logger.warning(
            "n_decoding_steps not provided in acoustic_transformer_args, defaulting to 7. "
            "Please add 'n_decoding_steps' to params.json under acoustic_transformer_args."
        )
        acoustic_args["n_decoding_steps"] = 7

    return {
        "sampling_rate": encoder_args["audio_encoding_args"]["sampling_rate"],
        "codec_args": audio_tokenizer_args,
        "audio_model_args": encoder_args,
        "speaker_id": (audio_tokenizer_args or {}).get("voice", {}),
    }


def _parse_voxtral_tts(config_dict: dict) -> tuple[dict, PretrainedConfig]:
    from vllm.transformers_utils.configs.mistral import (
        _remap_general_mistral_args,
        _remap_mistral_quantization_args,
    )

    audio_config: dict[str, Any] = {}
    if (config_dict.get("multimodal") or {}).get("audio_model_args"):
        audio_config = _remap_voxtral_tts_audio_args(config_dict)

    text_config = {k: v for k, v in config_dict.items() if k != "multimodal"}
    text_config = _remap_general_mistral_args(text_config)
    if text_config.get("quantization"):
        text_config = _remap_mistral_quantization_args(text_config)
    text_config.setdefault("architectures", ["MistralForCausalLM"])

    config = VoxtralTTSConfig(
        text_config=PretrainedConfig.from_dict(text_config),
        audio_config=audio_config,
        architectures=config_dict.get("architectures", ["VoxtralTTSForConditionalGeneration"]),
    )
    return config_dict, config


class VoxtralTTSConfigParser(MistralConfigParser):
    """Mistral parser that also recognizes Voxtral-TTS checkpoints."""

    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, PretrainedConfig]:
        config_dict = _download_mistral_config_file(model, revision)

        if _is_voxtral_tts_params(config_dict):
            return _parse_voxtral_tts(config_dict)

        return super().parse(
            model,
            trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )


# Replace the default "mistral" slot directly.
# Any non-Voxtral-TTS Mistral ckpt still goes through
# the upstream code path via super().parse().
_CONFIG_FORMAT_TO_CONFIG_PARSER["mistral"] = VoxtralTTSConfigParser

__all__ = ["VoxtralTTSConfigParser"]
