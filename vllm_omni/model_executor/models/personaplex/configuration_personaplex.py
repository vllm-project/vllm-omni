# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for PersonaPlex (a Moshi finetune; 2-stage audio->audio pipeline).

PersonaPlex is a staged AR speech model composed of:

* a *temporal transformer* (a Llama-variant Helium backbone) that predicts, per
  frame, the next text token and the codebook-0 audio token;
* a *depformer* (a small per-step transformer) that, conditioned on the temporal
  hidden state, predicts the remaining audio codebooks;
* the Mimi neural audio codec, which turns the audio codebooks into 24 kHz PCM.

The temporal-transformer hyperparameters live in
:class:`~vllm_omni.model_executor.models.personaplex.configuration_helium.HeliumConfig`
(already used by the talker the lead is building). This module reuses it as the
``temporal_config`` sub-config so the config tree carries a single source of
truth, and adds two further sub-configs (``depformer_config`` and
``mimi_config``).

All defaults are *measured* from the PersonaPlex checkpoint, whose own
``config.json`` is empty; do not treat the Moshi higher-level kwargs as
authoritative.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from vllm_omni.model_executor.models.personaplex.configuration_helium import (
    HeliumConfig,
)

logger = logging.get_logger(__name__)

__all__ = [
    "PersonaPlexConfig",
    "PersonaPlexDepformerConfig",
    "PersonaPlexMimiConfig",
    "HeliumConfig",
]


class PersonaPlexDepformerConfig(PretrainedConfig):
    r"""Configuration for the PersonaPlex depformer (per-step code predictor).

    The depformer is a small autoregressive transformer that runs ``dep_q``
    inner steps per temporal frame, conditioned on the temporal hidden state,
    to predict the audio codebooks. Only the first ``num_active_codebooks``
    codebooks (``cb 0..7``) are decoded to PCM by Mimi; the remaining codebooks
    up to ``dep_q`` are predicted but not vocoded.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the depformer hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of depformer transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads per depformer layer.
        head_dim (`int`, *optional*, defaults to 64):
            Per-head attention dimension (``hidden_size // num_attention_heads``).
        max_position_embeddings (`int`, *optional*, defaults to 8):
            The depformer context length (``ctx``): one position per inner step.
        dep_q (`int`, *optional*, defaults to 16):
            Number of audio codebooks the depformer predicts per frame.
        num_active_codebooks (`int`, *optional*, defaults to 8):
            Number of leading codebooks actually decoded to PCM by Mimi.
        card (`int`, *optional*, defaults to 2048):
            Per-codebook cardinality (audio vocab size).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rms_norm_eps (`float`, *optional*, defaults to 1e-8):
            The epsilon used by the fp32 RMS normalization layers.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            SwiGLU gate activation.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether attention projections carry a bias.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether the MLP projections carry a bias.
    """

    model_type = "personaplex_depformer"
    keys_to_ignore_at_inference = ("past_key_values",)

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,
        head_dim: int = 64,
        num_key_value_heads: int = 16,
        intermediate_size: int = 2816,
        max_position_embeddings: int = 8,
        dep_q: int = 16,
        num_active_codebooks: int = 8,
        card: int = 2048,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-8,
        hidden_act: str = "silu",
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dep_q = dep_q
        self.num_active_codebooks = num_active_codebooks
        self.card = card
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_cache = use_cache


class PersonaPlexMimiConfig(PretrainedConfig):
    r"""Configuration for the Mimi neural audio codec used by PersonaPlex.

    The Mimi weights and module live in the external ``moshi`` package; this
    config only carries the scalars the vllm-omni serving layer needs to size
    buffers and compute audio durations. The actual decoder is instantiated by
    :class:`PersonaPlexCode2Wav` via ``moshi.models.loaders.get_mimi``.

    Args:
        sample_rate (`int`, *optional*, defaults to 24000):
            Output PCM sample rate in Hz.
        frame_rate (`float`, *optional*, defaults to 12.5):
            Mimi codec frame rate in Hz (one frame every 80 ms).
        samples_per_frame (`int`, *optional*, defaults to 1920):
            PCM samples produced per codec frame (``sample_rate / frame_rate``).
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of active audio codebooks decoded to PCM (``cb 0..7``).
        card (`int`, *optional*, defaults to 2048):
            Per-codebook cardinality.
        num_channels (`int`, *optional*, defaults to 1):
            Number of output audio channels (mono).
        mimi_name (`str`, *optional*):
            Filename of the Mimi weight checkpoint inside the model repo. When
            ``None``, :class:`PersonaPlexCode2Wav` falls back to
            ``moshi.models.loaders.MIMI_NAME``.
    """

    model_type = "personaplex_mimi"

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_rate: float = 12.5,
        samples_per_frame: int = 1920,
        num_codebooks: int = 8,
        card: int = 2048,
        num_channels: int = 1,
        mimi_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.samples_per_frame = samples_per_frame
        self.num_codebooks = num_codebooks
        self.card = card
        self.num_channels = num_channels
        self.mimi_name = mimi_name


class PersonaPlexConfig(PretrainedConfig):
    r"""Top-level configuration for ``PersonaPlexTalkerForConditionalGeneration``.

    Mirrors the Qwen3-TTS config layout: a top-level config holding sub-configs
    for each component. The temporal transformer config is the *text config*
    that vLLM consumes (it exposes ``hidden_size`` / ``num_attention_heads`` and
    drives the talker's vLLM Llama backbone).

    Args:
        temporal_config (`dict` or `HeliumConfig`, *optional*):
            The temporal-transformer (Helium) backbone config.
        depformer_config (`dict` or `PersonaPlexDepformerConfig`, *optional*):
            The depformer (per-step code predictor) config.
        mimi_config (`dict` or `PersonaPlexMimiConfig`, *optional*):
            The Mimi codec config.
        text_vocab_size (`int`, *optional*, defaults to 32000):
            Text / ``lm_head`` vocabulary size.
        text_embedding_rows (`int`, *optional*, defaults to 32001):
            Number of rows in the text embedding table (one extra padding row).
        audio_vocab_size (`int`, *optional*, defaults to 2048):
            Per-codebook audio cardinality (``card``).
        num_audio_codebooks (`int`, *optional*, defaults to 16):
            Total number of audio codebooks (``n_q``).
        mimi_name (`str`, *optional*):
            Convenience mirror of ``mimi_config.mimi_name``; if set, it
            overrides the value carried inside ``mimi_config``.
    """

    model_type = "personaplex"
    sub_configs = {
        "temporal_config": HeliumConfig,
        "depformer_config": PersonaPlexDepformerConfig,
        "mimi_config": PersonaPlexMimiConfig,
    }

    def __init__(
        self,
        temporal_config: dict | HeliumConfig | None = None,
        depformer_config: dict | PersonaPlexDepformerConfig | None = None,
        mimi_config: dict | PersonaPlexMimiConfig | None = None,
        text_vocab_size: int = 32000,
        text_embedding_rows: int = 32001,
        audio_vocab_size: int = 2048,
        num_audio_codebooks: int = 16,
        mimi_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        if temporal_config is None:
            temporal_config = {}
            logger.info("temporal_config is None. Initializing Helium backbone with default values")
        if depformer_config is None:
            depformer_config = {}
            logger.info("depformer_config is None. Initializing depformer with default values")
        if mimi_config is None:
            mimi_config = {}
            logger.info("mimi_config is None. Initializing Mimi codec config with default values")

        self.temporal_config = self._coerce(temporal_config, HeliumConfig)
        self.depformer_config = self._coerce(depformer_config, PersonaPlexDepformerConfig)
        self.mimi_config = self._coerce(mimi_config, PersonaPlexMimiConfig)

        super().__init__(**kwargs)

        # The PersonaPlex checkpoint's config.json is empty (no ``architectures``);
        # supply the primary (stage-0 talker) arch so vLLM's ModelConfig resolves a
        # generate-capable model. The code2wav stage arch comes from the pipeline.
        if not getattr(self, "architectures", None):
            self.architectures = ["PersonaPlexTalkerForConditionalGeneration"]

        self.text_vocab_size = text_vocab_size
        self.text_embedding_rows = text_embedding_rows
        self.audio_vocab_size = audio_vocab_size
        self.num_audio_codebooks = num_audio_codebooks

        # ``mimi_name`` is allowed both at the top level (convenient for
        # hf_overrides / a flat config.json) and inside ``mimi_config``. Keep
        # them in sync, preferring an explicit top-level value.
        if mimi_name is not None:
            self.mimi_config.mimi_name = mimi_name
        self.mimi_name = self.mimi_config.mimi_name

    @staticmethod
    def _coerce(value: Any, config_cls: type[PretrainedConfig]) -> PretrainedConfig:
        if isinstance(value, config_cls):
            return value
        if isinstance(value, dict):
            return config_cls(**value)
        raise TypeError(f"Expected dict or {config_cls.__name__}, got {type(value).__name__}")

    @property
    def sample_rate(self) -> int:
        """Output PCM sample rate in Hz (delegates to the Mimi config)."""
        return int(self.mimi_config.sample_rate)

    def get_text_config(self, **kwargs: Any) -> PretrainedConfig:
        # vLLM expects the text config to expose hidden_size / num_attention_heads.
        # For PersonaPlex the temporal-transformer (Helium) config is the text
        # model config that drives the talker's vLLM Llama backbone.
        return self.temporal_config
