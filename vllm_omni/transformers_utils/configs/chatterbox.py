# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox TTS configuration.

The Hub repos ship no ``config.json``, so every constant lives here. Values
come from upstream ``chatterbox`` 0.1.7: ``GPT2_MEDIUM_CONFIG`` and
``LLAMA_520M_CONFIG_DICT`` in ``models/t3/llama_configs.py``, the ``T3Config``
overrides in ``ChatterboxTurboTTS.from_local`` and ``ChatterboxTTS.from_local``,
and ``models/s3gen``.

``variant`` selects the checkpoint. Everything that differs between the two
checkpoints is a property of the variant, so the deploy file's
``hf_overrides: {variant: original}`` is enough however it arrives: at
construction from a config.json, or afterwards through ``update``.
"""

import operator

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

# Control ids shared by every checkpoint: the start-of-speech and stop tokens.
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
# Request-id suffix of a guided request's unconditional companion.
CFG_UNCOND_SUFFIX = "__cfg_uncond"

FLOW_DTYPES = frozenset({"float32", "bfloat16"})

# What differs between the checkpoints. The speech vocabulary sizes the
# sampler (``vocab_size``): no tokenizer is loaded for these stages, and the
# text vocabulary only sizes the text embedding and the backbone's idle
# ``wte``.
VARIANT_FIELDS: dict[str, dict[str, object]] = {
    "turbo": {
        "vocab_size": 6563,
        "speech_vocab_size": 6563,
        "text_vocab_size": 50276,
        "num_hidden_layers": 24,
        "max_position_embeddings": 8196,
        "activation_function": "gelu_new",
        "cond_prompt_len": 375,
        "enc_cond_seconds": 15,
        "min_ref_seconds": 5.0,
        "loudness_target_lufs": -27.0,
        "n_silence_tokens": 3,
        "meanflow": True,
        "n_cfm_timesteps": 2,
        "t3_weights": "t3_turbo_v1.safetensors",
        "s3gen_weights": "s3gen_meanflow.safetensors",
    },
    "original": {
        "vocab_size": 8194,
        "speech_vocab_size": 8194,
        "text_vocab_size": 704,
        "num_hidden_layers": 30,
        "max_position_embeddings": 131072,
        "activation_function": "silu",
        "cond_prompt_len": 150,
        "enc_cond_seconds": 6,
        "min_ref_seconds": 0.0,
        "loudness_target_lufs": None,
        "n_silence_tokens": 0,
        "meanflow": False,
        "n_cfm_timesteps": 10,
        "t3_weights": "t3_cfg.safetensors",
        "s3gen_weights": "s3gen.safetensors",
    },
}


class ChatterboxConfig(PretrainedConfig):
    """Constants for both Chatterbox stages of one checkpoint."""

    model_type = "chatterbox"

    def __init__(self, variant: str = "turbo", flow_dtype: str = "bfloat16", **kwargs):
        if variant not in VARIANT_FIELDS:
            raise ValueError("Chatterbox variant must be 'turbo' or 'original'")
        kwargs.setdefault("eos_token_id", STOP_SPEECH_TOKEN)
        super().__init__(**kwargs)
        self.variant = variant
        # Stage 1's flow runs under bfloat16 autocast unless a deploy file
        # overrides it; HiFT always runs in float32. Stage 1 validates it
        # when it is built.
        self.flow_dtype = flow_dtype

        # Ids at or above this are control tokens the decoder must never see.
        self.speech_token_limit = START_SPEECH_TOKEN
        self.start_speech_token = START_SPEECH_TOKEN
        self.stop_speech_token = STOP_SPEECH_TOKEN

        # T3 backbone (GPT-2 medium or LLaMA 520M: same width, heads and MLP).
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.num_key_value_heads = 16
        self.intermediate_size = 4096
        self.layer_norm_epsilon = 1e-5
        # Upstream's generation cap for both checkpoints: ``inference_turbo``'s
        # ``max_gen_len`` and ``generate``'s ``max_new_tokens``.
        self.max_new_tokens = 1000

        # Reference conditioning.
        self.speaker_embed_size = 256
        self.dec_cond_seconds = 10
        self.s3_tokenizer_name = "speech_tokenizer_v2_25hz"

        # Speech tokens and audio.
        self.s3_sample_rate = 16000
        self.sample_rate = 24000
        self.token_rate = 25
        self.token_mel_ratio = 2
        self.pre_lookahead_len = 3
        self.mel = {
            "n_fft": 1920,
            "num_mels": 80,
            "sampling_rate": 24000,
            "hop_size": 480,
            "win_size": 1920,
            "fmin": 0,
            "fmax": 8000,
            "center": False,
        }
        self.silence_token = 4299
        self.ve_weights = "ve.safetensors"

    def variant_field(self, name: str) -> object:
        """The checkpoint-dependent value of ``name`` for the current variant.

        Raises:
            ValueError: If ``variant`` was overridden with an unknown value.
        """
        fields = VARIANT_FIELDS.get(self.variant)
        if fields is None:
            raise ValueError(f"Chatterbox variant must be 'turbo' or 'original', got {self.variant!r}")
        return fields[name]


for _name in VARIANT_FIELDS["turbo"]:
    setattr(ChatterboxConfig, _name, property(operator.methodcaller("variant_field", _name)))

AutoConfig.register("chatterbox", ChatterboxConfig)

__all__ = ["ChatterboxConfig"]
