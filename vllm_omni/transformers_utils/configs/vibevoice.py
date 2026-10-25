# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice config for the native HF-schema checkpoint (vibevoice/VibeVoice-1.5B-hf).

The project's Transformers 5.13/5.14 dependency lacks the full TTS config, so
this shim provides the native schema without loading remote code. Only the
native sub-config layout is accepted: ``audio_config``,
``semantic_model_config``, ``text_config`` and ``diffusion_head_config``.
The original Microsoft schema and the early fork layout are not converted.
"""

from typing import Any

from transformers import AutoConfig, PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


def _sub_config(value: dict[str, Any] | PreTrainedConfig | None, model_type: str, **defaults: Any) -> PreTrainedConfig:
    if isinstance(value, PreTrainedConfig):
        return value
    value = dict(defaults) if value is None else dict(value)
    return CONFIG_MAPPING[value.pop("model_type", model_type)](**value)


class VibeVoiceDiffusionHeadConfig(PreTrainedConfig):
    """Native-schema diffusion head sub-config (no model_type lookup)."""

    model_type = "vibevoice_diffusion"

    def __init__(
        self,
        hidden_size: int = 1536,
        latent_size: int = 64,
        num_hidden_layers: int = 4,
        intermediate_size: int = 4608,
        rms_norm_eps: float = 1e-5,
        hidden_act: str = "silu",
        frequency_embedding_size: int = 256,
        diffusion_max_period: int = 10000,
        mlp_bias: bool = False,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.frequency_embedding_size = frequency_embedding_size
        self.diffusion_max_period = diffusion_max_period
        self.mlp_bias = mlp_bias
        super().__init__(**kwargs)


class VibeVoiceConfig(PreTrainedConfig):
    """Root config for the native VibeVoice checkpoint."""

    model_type = "vibevoice"

    def __init__(
        self,
        audio_config: dict[str, Any] | PreTrainedConfig | None = None,
        semantic_model_config: dict[str, Any] | PreTrainedConfig | None = None,
        text_config: dict[str, Any] | PreTrainedConfig | None = None,
        diffusion_head_config: dict[str, Any] | VibeVoiceDiffusionHeadConfig | None = None,
        pad_token_id: int = 151643,
        eos_token_id: int = 151643,
        audio_bos_token_id: int = 151652,
        audio_eos_token_id: int = 151653,
        audio_token_id: int = 151654,
        **kwargs: Any,
    ) -> None:
        if {
            "acoustic_tokenizer_config",
            "semantic_tokenizer_config",
            "decoder_config",
            "diffusion_config",
        } & kwargs.keys():
            raise ValueError("VibeVoice only supports the native HF schema; legacy fields are not accepted.")
        self.audio_config = _sub_config(audio_config, "vibevoice_acoustic_tokenizer")
        self.semantic_model_config = _sub_config(
            semantic_model_config, "vibevoice_acoustic_tokenizer_encoder", hidden_size=128
        )
        self.text_config = _sub_config(text_config, "qwen2")

        if isinstance(diffusion_head_config, dict):
            diffusion_head_config = dict(diffusion_head_config)
            # Native checkpoints ship an empty model_type here; field defaults
            # come from the dataclass-style constructor, not CONFIG_MAPPING.
            diffusion_head_config.pop("model_type", None)
            diffusion_head_config = VibeVoiceDiffusionHeadConfig(**diffusion_head_config)
        elif diffusion_head_config is None:
            diffusion_head_config = VibeVoiceDiffusionHeadConfig(
                hidden_size=self.text_config.hidden_size,
                latent_size=self.audio_config.hidden_size,
            )
        self.diffusion_head_config = diffusion_head_config

        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_token_id = audio_token_id

        if self.diffusion_head_config.hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"diffusion_head_config.hidden_size ({self.diffusion_head_config.hidden_size}) must match "
                f"text_config.hidden_size ({self.text_config.hidden_size})."
            )
        if self.diffusion_head_config.latent_size != self.audio_config.hidden_size:
            raise ValueError(
                f"diffusion_head_config.latent_size ({self.diffusion_head_config.latent_size}) must match "
                f"audio_config.hidden_size ({self.audio_config.hidden_size})."
            )

        self.vocab_size = self.text_config.vocab_size
        self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        return int(self.text_config.hidden_size)


# Do not replace a native registration if Transformers already provides it.
if "vibevoice" not in CONFIG_MAPPING:
    AutoConfig.register("vibevoice", VibeVoiceConfig)

__all__ = ["VibeVoiceConfig", "VibeVoiceDiffusionHeadConfig"]
