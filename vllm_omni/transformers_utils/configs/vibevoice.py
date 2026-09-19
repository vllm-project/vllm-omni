# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# SPDX-FileCopyrightText: Copyright 2026 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
#
# Adapted from Hugging Face Transformers PR #40546,
# src/transformers/models/vibevoice/configuration_vibevoice.py.
"""VibeVoice TTS config shim with an HF-schema runtime representation.

Microsoft's original checkpoint is the public input. Its nested config schema
is normalized at load time to the representation used by Transformers PR
#40546: ``audio_config``, ``semantic_model_config``, ``text_config``, and flat
diffusion fields. Weight-name conversion remains a model-loader responsibility.
See ``docs/models/vibevoice.md`` for supported deployment behavior.
"""

from copy import deepcopy
from typing import Any

from huggingface_hub.dataclasses import strict
from transformers import AutoConfig, PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.vibevoice_acoustic_tokenizer.configuration_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerConfig as _VibeVoiceAcousticTokenizerConfig,
)
from transformers.models.vibevoice_acoustic_tokenizer.configuration_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerEncoderConfig as _VibeVoiceAcousticTokenizerEncoderConfig,
)
from transformers.utils import auto_docstring

_ORIGINAL_TOKENIZER_CONFIG_KEYS_TO_REMOVE = {
    "decoder_depths",
    "decoder_n_filters",
    "decoder_ratios",
    "std_dist_type",
    "pad_mode",
    "conv_bias",
    "causal",
    "mixer_layer",
    "layernorm",
    "disable_last_norm",
    "conv_norm",
    "corpus_normalize",
    "layernorm_elementwise_affine",
}
_MISSING = object()


def _convert_original_tokenizer_config(
    config: dict[str, Any],
    *,
    model_type: str,
    acoustic: bool,
) -> dict[str, Any]:
    """Convert a Microsoft tokenizer config to an upstream HF child config."""
    converted = deepcopy(config)
    encoder_depths = converted.pop("encoder_depths", None)
    if isinstance(encoder_depths, str):
        encoder_depths = [int(depth) for depth in encoder_depths.split("-")]
    if encoder_depths is not None:
        converted["depths"] = encoder_depths

    if "encoder_ratios" in converted:
        converted["downsampling_ratios"] = list(reversed(converted.pop("encoder_ratios")))
    if "encoder_n_filters" in converted:
        converted["num_filters"] = converted.pop("encoder_n_filters")
    if "layernorm_eps" in converted:
        converted["rms_norm_eps"] = converted.pop("layernorm_eps")
    if "vae_dim" in converted:
        converted["hidden_size"] = converted.pop("vae_dim")

    # HF's acoustic tokenizer incorporates the original hard-coded 0.8
    # latent scaling into vae_std. The semantic encoder uses the upstream
    # default because it does not sample acoustic VAE noise.
    fix_std = converted.pop("fix_std", None)
    if acoustic and fix_std is not None:
        converted["vae_std"] = fix_std / 0.8

    # Keep weight_init_value as checkpoint metadata while making the actual
    # initializer field explicit, matching the converted HF checkpoint.
    if "weight_init_value" in converted:
        converted.setdefault("initializer_range", converted["weight_init_value"])

    for key in _ORIGINAL_TOKENIZER_CONFIG_KEYS_TO_REMOVE:
        converted.pop(key, None)
    converted["model_type"] = model_type
    return converted


def _convert_original_diffusion_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten Microsoft diffusion fields as consumed by PR #40546."""
    converted = deepcopy(config)
    hidden_size = converted.pop("hidden_size", 1536)
    ffn_ratio = converted.pop("head_ffn_ratio", 3.0)
    converted["intermediate_size"] = converted.get("intermediate_size", int(ffn_ratio * hidden_size))
    converted["num_head_layers"] = converted.pop("head_layers", converted.get("num_head_layers", 4))
    if converted.get("ddpm_beta_schedule") == "cosine":
        converted["ddpm_beta_schedule"] = "squaredcos_cap_v2"
    for key in ("speech_vae_dim", "diffusion_type", "ddpm_batch_mul", "latent_size", "model_type"):
        converted.pop(key, None)
    return converted


@auto_docstring(checkpoint="bezzam/VibeVoice-1.5B-hf")
@strict
class VibeVoiceConfig(PreTrainedConfig):
    r"""
    semantic_model_config (`Union[AutoConfig, dict]`, *optional*):
        The config object or dictionary of the semantic tokenizer encoder. This tokenizer extracts semantic features
        from audio.
    audio_bos_token_id (`int`, *optional*, defaults to 151652):
        The token ID indicating the start of audio tokens.
    audio_eos_token_id (`int`, *optional*, defaults to 151653):
        The token ID indicating the end of audio tokens.
    num_head_layers (`int`, *optional*, defaults to 4):
        Number of layers in the diffusion head.
    frequency_embedding_size (`int`, *optional*, defaults to 256):
        The size of the sinusoidal frequency embedding for timestep encoding in the diffusion head.
    diffusion_max_period (`int`, *optional*, defaults to 10000):
        The maximum period for the sinusoidal frequency embedding in the diffusion head.
    """

    model_type = "vibevoice"
    sub_configs = {
        "audio_config": AutoConfig,
        "semantic_model_config": AutoConfig,
        "text_config": AutoConfig,
    }

    # Keep the plan aligned with the module names used by PR #40546.
    base_model_tp_plan = {
        "language_model.layers.*.self_attn.q_proj": "colwise",
        "language_model.layers.*.self_attn.k_proj": "colwise",
        "language_model.layers.*.self_attn.v_proj": "colwise",
        "language_model.layers.*.self_attn.o_proj": "rowwise",
        "language_model.layers.*.mlp.gate_proj": "colwise",
        "language_model.layers.*.mlp.up_proj": "colwise",
        "language_model.layers.*.mlp.down_proj": "rowwise",
    }

    audio_config: dict[str, Any] | PreTrainedConfig | None = None
    semantic_model_config: dict[str, Any] | PreTrainedConfig | None = None
    text_config: dict[str, Any] | PreTrainedConfig | None = None
    pad_token_id: int = 151643
    eos_token_id: int = 151643
    audio_bos_token_id: int = 151652
    audio_eos_token_id: int = 151653
    audio_token_id: int = 151654
    num_head_layers: int = 4
    intermediate_size: int = 4608
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    frequency_embedding_size: int = 256
    diffusion_max_period: int = 10000
    mlp_bias: bool = False

    def _normalize_original_schema(self, kwargs: dict[str, Any]) -> None:
        """Convert Microsoft root-config aliases to the HF runtime schema."""
        acoustic_config = kwargs.pop("acoustic_tokenizer_config", _MISSING)
        semantic_config = kwargs.pop("semantic_tokenizer_config", _MISSING)
        decoder_config = kwargs.pop("decoder_config", _MISSING)
        diffusion_config = kwargs.pop("diffusion_head_config", _MISSING)

        if acoustic_config is not _MISSING:
            if self.audio_config is not None:
                raise ValueError("Cannot provide both `acoustic_tokenizer_config` and `audio_config`.")
            self.audio_config = _convert_original_tokenizer_config(
                acoustic_config or {},
                model_type="vibevoice_acoustic_tokenizer",
                acoustic=True,
            )

        if semantic_config is not _MISSING:
            if self.semantic_model_config is not None:
                raise ValueError("Cannot provide both `semantic_tokenizer_config` and `semantic_model_config`.")
            self.semantic_model_config = _convert_original_tokenizer_config(
                semantic_config or {},
                model_type="vibevoice_acoustic_tokenizer_encoder",
                acoustic=False,
            )

        if decoder_config is not _MISSING:
            if self.text_config is not None:
                raise ValueError("Cannot provide both `decoder_config` and `text_config`.")
            self.text_config = deepcopy(decoder_config or {})
            self.text_config.setdefault("model_type", "qwen2")
            if "torch_dtype" in self.text_config and "dtype" not in self.text_config:
                self.text_config["dtype"] = self.text_config.pop("torch_dtype")

        if diffusion_config is not _MISSING:
            flattened = _convert_original_diffusion_config(diffusion_config or {})
            for name, value in flattened.items():
                if name in type(self).__annotations__:
                    setattr(self, name, value)
                else:
                    kwargs[name] = value

        # Normalize the deprecated root dtype alias as well as the alias in
        # decoder_config above. This matches the HF conversion script and
        # avoids a warning from PreTrainedConfig for every official load.
        if "torch_dtype" in kwargs:
            kwargs.setdefault("dtype", kwargs["torch_dtype"])
            kwargs.pop("torch_dtype")

        # These original derived fields are represented by the normalized
        # child configs and must not leak into serialized HF-schema output.
        kwargs.pop("acoustic_vae_dim", None)
        kwargs.pop("semantic_vae_dim", None)

    def __post_init__(self, **kwargs: Any) -> None:
        self._normalize_original_schema(kwargs)

        # Copy caller-owned dictionaries (including nested lists) before adding
        # model_type or handing them to child config constructors.
        if isinstance(self.audio_config, dict):
            audio_config = deepcopy(self.audio_config)
            audio_config["model_type"] = audio_config.get("model_type", "vibevoice_acoustic_tokenizer")
            self.audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["vibevoice_acoustic_tokenizer"]()

        if isinstance(self.semantic_model_config, dict):
            semantic_model_config = deepcopy(self.semantic_model_config)
            semantic_model_config["model_type"] = semantic_model_config.get(
                "model_type", "vibevoice_acoustic_tokenizer_encoder"
            )
            self.semantic_model_config = CONFIG_MAPPING[semantic_model_config["model_type"]](**semantic_model_config)
        elif self.semantic_model_config is None:
            self.semantic_model_config = CONFIG_MAPPING["vibevoice_acoustic_tokenizer_encoder"](hidden_size=128)

        if isinstance(self.text_config, dict):
            text_config = deepcopy(self.text_config)
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        self.vocab_size = self.text_config.vocab_size
        self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)
        super().__post_init__(**kwargs)

    @property
    def hidden_size(self) -> int:
        """Hidden size consumed by the PR #40546 diffusion head."""
        return int(getattr(self.text_config, "hidden_size"))


# Ensure the Acoustic Tokenizer sub-config model types are registered in
# CONFIG_MAPPING. Some Transformers versions in the >=5.10.1,<5.15 range
# ship the submodules but do not register the plain "vibevoice_acoustic_tokenizer"
# key in CONFIG_MAPPING. Register from the submodule classes so the shim works
# across the full declared range.
for _model_type, _config_cls in (
    ("vibevoice_acoustic_tokenizer", _VibeVoiceAcousticTokenizerConfig),
    ("vibevoice_acoustic_tokenizer_encoder", _VibeVoiceAcousticTokenizerEncoderConfig),
):
    if _model_type not in CONFIG_MAPPING:
        AutoConfig.register(_model_type, _config_cls)

# Register the top-level config only if Transformers hasn't already added a
# built-in VibeVoiceConfig. Once PR #40546 fully lands, this shim should be
# removed in favor of the upstream class.
if "vibevoice" not in CONFIG_MAPPING:
    AutoConfig.register("vibevoice", VibeVoiceConfig)


__all__ = ["VibeVoiceConfig"]
