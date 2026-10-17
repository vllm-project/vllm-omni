# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration classes for Audio8 TTS Preview (``model_type = "arktts"``).

Audio8's HuggingFace config is flat and uses DualAR field names (``dim``,
``n_layer``, ``n_head``, ``fast_dim``, ...); this module re-exports them under
Qwen2 attribute names so vLLM's ``Qwen2Model`` consumes the Slow AR half, and
splits the Fast AR half into its own sub-config. The backbone is Qwen2 rather
than Qwen3 (Fish Speech) because Audio8 uses ``qkv_bias=true`` with no q/k
norm -- exactly the Qwen2 attention shape.
"""

from __future__ import annotations

from typing import Any

from transformers import FalconH1Config, PretrainedConfig

#: Flat ``arktts`` fields forwarded verbatim into ``FalconH1Config`` when the
#: checkpoint declares ``slow_backbone == "falcon_h1"``. The names are already
#: shared between the two configs (verified against the 0.1b ``config.json``).
_HYBRID_PASSTHROUGH_FIELDS = (
    "attention_bias",
    "attention_dropout",
    "attention_in_multiplier",
    "attention_o_bias",
    "attention_out_multiplier",
    "attention_qk_norm",
    "attention_qkv_bias",
    "bos_token_id",
    "expansion_factor",
    "initializer_range",
    "key_multiplier",
    "embedding_multiplier",
    "lm_head_multiplier",
    "mamba_chunk_size",
    "mamba_conv_bias",
    "mamba_d_conv",
    "mamba_d_head",
    "mamba_d_ssm",
    "mamba_d_state",
    "mamba_expand",
    "mamba_n_groups",
    "mamba_n_heads",
    "mamba_norm_before_gate",
    "mamba_proj_bias",
    "mamba_rms_norm",
    "mamba_use_mlp",
    "mlp_bias",
    "mlp_multipliers",
    "projectors_bias",
    "ssm_in_multiplier",
    "ssm_multipliers",
    "ssm_out_multiplier",
    "time_step_floor",
    "time_step_max",
    "time_step_min",
    "time_step_rank",
    "use_cache",
)

#: Codec frame stride in waveform samples (encoder rates 2*4*8*8 = 512 times the
#: quantizer's 2*2 downsample). 44100 / 2048 ~= 21.5 frames per second.
ARKTTS_CODEC_FRAME_SIZE = 2048
ARKTTS_CODEC_SAMPLE_RATE = 44100
#: Only the first codebook is a 4096-entry semantic codebook; codebooks 1..N-1
#: are 1024-entry residual codebooks (see ``ArkttsDownsampleQuantizer``).
ARKTTS_SEMANTIC_CODEBOOK_SIZE = 4096
ARKTTS_RESIDUAL_CODEBOOK_SIZE = 1024


class Audio8TTSSlowARConfig(PretrainedConfig):
    """Slow AR config exposed with Qwen2-compatible attribute names."""

    model_type = "arktts_slow_ar"

    def __init__(
        self,
        vocab_size: int = 155776,
        dim: int = 896,
        n_head: int = 14,
        n_local_heads: int = 2,
        head_dim: int = 64,
        n_layer: int = 24,
        intermediate_size: int = 4864,
        max_seq_len: int = 2048,
        rope_base: float = 1_000_000.0,
        norm_eps: float = 1e-6,
        attention_qkv_bias: bool = True,
        attention_qk_norm: bool = False,
        tie_word_embeddings: bool = True,
        codebook_size: int = ARKTTS_SEMANTIC_CODEBOOK_SIZE,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151678,
        semantic_end_id: int = 155773,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        **kwargs: Any,
    ) -> None:
        # Audio8 field names -> standard Transformers / Qwen2 names.
        self.hidden_size = int(dim)
        self.num_attention_heads = int(n_head)
        self.num_key_value_heads = int(n_local_heads)
        self.head_dim = int(head_dim)
        self.num_hidden_layers = int(n_layer)
        self.intermediate_size = int(intermediate_size)
        self.max_position_embeddings = int(max_seq_len)
        self.rms_norm_eps = float(norm_eps)
        self.hidden_act = "silu"
        # Qwen2Attention hardcodes qkv bias=True / o_proj bias=False, which is
        # what Audio8 TTS ships. Keep the flags so a future checkpoint that
        # flips them fails loudly in the model rather than loading silently.
        self.attention_qkv_bias = bool(attention_qkv_bias)
        self.attention_qk_norm = bool(attention_qk_norm)
        # Set explicitly (rather than relying on vLLM's set_default_rope_theta)
        # so that a checkpoint with a non-default rope_base is honoured.
        self.rope_parameters = {"rope_type": "default", "rope_theta": float(rope_base)}

        # Codec / codebook fields.
        self.codebook_size = int(codebook_size)
        self.num_codebooks = int(num_codebooks)
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)

        super().__init__(
            vocab_size=int(vocab_size),
            tie_word_embeddings=bool(tie_word_embeddings),
            eos_token_id=int(eos_token_id),
            pad_token_id=int(pad_token_id),
            **kwargs,
        )


class Audio8TTSHybridSlowARConfig(FalconH1Config):
    """Slow AR config for the hybrid-Mamba2 checkpoint variant (0.1b).

    Subclasses :class:`transformers.FalconH1Config` so the backbone can be
    handed to vLLM's native ``FalconH1ForCausalLM`` unchanged -- the same
    wrapping pattern ``nemotron_voicechat_thinker`` uses for
    ``NemotronHForCausalLM`` -- while still carrying the ``arktts``-specific
    fields the stage logic reads.

    SCAFFOLD: not yet exercised on hardware. See the design note on issue
    #6495 for the open parity items (muP scalars, RoPE style, compact head).
    """

    model_type = "arktts_slow_ar_hybrid"

    def __init__(
        self,
        *,
        codebook_size: int = ARKTTS_SEMANTIC_CODEBOOK_SIZE,
        num_codebooks: int = 10,
        semantic_begin_id: int = 65537,
        semantic_end_id: int = 69632,
        eos_token_id: int = 228,
        pad_token_id: int = 0,
        norm_fastlayer_input: bool = False,
        **kwargs: Any,
    ) -> None:
        self.codebook_size = int(codebook_size)
        self.num_codebooks = int(num_codebooks)
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        # muP scalars are absent from the 0.6b config (treated as 1.0 there) and
        # non-trivial on the 0.1b; the audio8 stage code must apply both.
        self.embedding_multiplier = float(kwargs.get("embedding_multiplier", 1.0))
        self.lm_head_multiplier = float(kwargs.get("lm_head_multiplier", 1.0))
        # ``max_seq_len`` is the arktts spelling of the HF field.
        if "max_seq_len" in kwargs and "max_position_embeddings" not in kwargs:
            kwargs["max_position_embeddings"] = kwargs.pop("max_seq_len")
        kwargs.pop("slow_backbone", None)
        super().__init__(
            eos_token_id=int(eos_token_id),
            pad_token_id=int(pad_token_id),
            **kwargs,
        )


class Audio8TTSFastARConfig(PretrainedConfig):
    """Fast AR config: the ``n_fast_layer`` residual-codebook predictor."""

    model_type = "arktts_fast_ar"

    def __init__(
        self,
        codebook_size: int = ARKTTS_SEMANTIC_CODEBOOK_SIZE,
        num_codebooks: int = 10,
        fast_dim: int = 896,
        fast_n_head: int = 14,
        fast_n_local_heads: int = 2,
        fast_head_dim: int = 64,
        n_fast_layer: int = 4,
        fast_intermediate_size: int = 4864,
        fast_attention_qkv_bias: bool = False,
        fast_attention_qk_norm: bool = False,
        rope_base: float = 1_000_000.0,
        norm_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = int(fast_dim)
        self.num_attention_heads = int(fast_n_head)
        self.num_key_value_heads = int(fast_n_local_heads)
        self.head_dim = int(fast_head_dim)
        self.num_hidden_layers = int(n_fast_layer)
        self.intermediate_size = int(fast_intermediate_size)
        # The Fast AR sequence is [slow hidden state, code_0, ..., code_{N-1}].
        self.max_position_embeddings = int(num_codebooks)
        self.rms_norm_eps = float(norm_eps)
        self.hidden_act = "silu"
        self.attention_qkv_bias = bool(fast_attention_qkv_bias)
        self.attention_qk_norm = bool(fast_attention_qk_norm)
        self.rope_theta = float(rope_base)
        self.num_codebooks = int(num_codebooks)

        super().__init__(vocab_size=int(codebook_size), **kwargs)


class Audio8TTSConfig(PretrainedConfig):
    """Top-level Audio8 TTS config (``model_type = "arktts"``).

    Accepts the flat HF checkpoint fields and derives ``text_config`` (Slow AR)
    and ``fast_ar_config`` (Fast AR).  ``get_text_config()`` returns the Slow AR
    config, which is what ``Qwen2Model`` reads.
    """

    model_type = "arktts"
    sub_configs = {
        "text_config": Audio8TTSSlowARConfig,
        "fast_ar_config": Audio8TTSFastARConfig,
    }

    def __init__(
        self,
        text_config: dict | Audio8TTSSlowARConfig | None = None,
        fast_ar_config: dict | Audio8TTSFastARConfig | None = None,
        slow_backbone: str | None = None,
        vocab_size: int = 155776,
        dim: int = 896,
        n_head: int = 14,
        n_local_heads: int = 2,
        head_dim: int = 64,
        n_layer: int = 24,
        intermediate_size: int = 4864,
        max_seq_len: int = 2048,
        rope_base: float = 1_000_000.0,
        norm_eps: float = 1e-6,
        attention_qkv_bias: bool = True,
        attention_qk_norm: bool = False,
        tie_word_embeddings: bool = True,
        codebook_size: int = ARKTTS_SEMANTIC_CODEBOOK_SIZE,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151678,
        semantic_end_id: int = 155773,
        n_fast_layer: int = 4,
        fast_dim: int = 896,
        fast_n_head: int = 14,
        fast_n_local_heads: int = 2,
        fast_head_dim: int = 64,
        fast_intermediate_size: int = 4864,
        fast_attention_qkv_bias: bool = False,
        fast_attention_qk_norm: bool = False,
        norm_fastlayer_input: bool = True,
        codec_filename: str = "codec.pth",
        codec_sample_rate: int = ARKTTS_CODEC_SAMPLE_RATE,
        codec_frame_size: int = ARKTTS_CODEC_FRAME_SIZE,
        codec_post_n_layer: int = 8,
        codec_post_n_head: int = 16,
        codec_post_n_local_heads: int = 8,
        codec_post_intermediate_size: int = 1216,
        ras_window_size: int = 10,
        ras_temperature: float = 1.0,
        ras_top_p: float = 0.9,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Audio8TTSSlowARConfig(**text_config)
        if isinstance(fast_ar_config, dict):
            fast_ar_config = Audio8TTSFastARConfig(**fast_ar_config)

        # The 0.1b checkpoint is flat with ``slow_backbone = "falcon_h1"``: build
        # the hybrid Slow AR config from the same flat fields. NOTE: the
        # ``sub_configs`` mapping above still points ``text_config`` at the dense
        # class, so a checkpoint that ships a *nested* text_config dict would
        # land in the dense class -- the 0.1b does not.
        if text_config is None and slow_backbone == "falcon_h1":
            text_config = Audio8TTSHybridSlowARConfig(
                vocab_size=vocab_size,
                hidden_size=dim,
                num_hidden_layers=n_layer,
                num_attention_heads=n_head,
                num_key_value_heads=n_local_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_seq_len,
                rms_norm_eps=norm_eps,
                rope_theta=rope_base,
                hidden_act="silu",
                codebook_size=codebook_size,
                num_codebooks=num_codebooks,
                semantic_begin_id=semantic_begin_id,
                semantic_end_id=semantic_end_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                **{k: kwargs[k] for k in _HYBRID_PASSTHROUGH_FIELDS if k in kwargs},
            )

        self.text_config = text_config or Audio8TTSSlowARConfig(
            vocab_size=vocab_size,
            dim=dim,
            n_head=n_head,
            n_local_heads=n_local_heads,
            head_dim=head_dim,
            n_layer=n_layer,
            intermediate_size=intermediate_size,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            norm_eps=norm_eps,
            attention_qkv_bias=attention_qkv_bias,
            attention_qk_norm=attention_qk_norm,
            tie_word_embeddings=tie_word_embeddings,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            semantic_begin_id=semantic_begin_id,
            semantic_end_id=semantic_end_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        self.fast_ar_config = fast_ar_config or Audio8TTSFastARConfig(
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            fast_dim=fast_dim,
            fast_n_head=fast_n_head,
            fast_n_local_heads=fast_n_local_heads,
            fast_head_dim=fast_head_dim,
            n_fast_layer=n_fast_layer,
            fast_intermediate_size=fast_intermediate_size,
            fast_attention_qkv_bias=fast_attention_qkv_bias,
            fast_attention_qk_norm=fast_attention_qk_norm,
            rope_base=rope_base,
            norm_eps=norm_eps,
        )

        # Fields the stages read directly off the top-level config.
        self.slow_backbone = str(slow_backbone) if slow_backbone is not None else None
        self.codebook_size = int(codebook_size)
        self.num_codebooks = int(num_codebooks)
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        self.norm_fastlayer_input = bool(norm_fastlayer_input)
        self.codec_filename = str(codec_filename)
        self.codec_sample_rate = int(codec_sample_rate)
        self.codec_frame_size = int(codec_frame_size)
        self.codec_post_n_layer = int(codec_post_n_layer)
        self.codec_post_n_head = int(codec_post_n_head)
        self.codec_post_n_local_heads = int(codec_post_n_local_heads)
        self.codec_post_intermediate_size = int(codec_post_intermediate_size)
        # Repetition-Aware Sampling (RAS): resample a repeated semantic token
        # from a flatter distribution instead of masking it.
        self.ras_window_size = int(ras_window_size)
        self.ras_temperature = float(ras_temperature)
        self.ras_top_p = float(ras_top_p)

        super().__init__(
            eos_token_id=int(eos_token_id),
            pad_token_id=int(pad_token_id),
            tie_word_embeddings=bool(tie_word_embeddings),
            **kwargs,
        )

    def get_text_config(self, *args: Any, **kwargs: Any) -> Audio8TTSSlowARConfig:
        return self.text_config


__all__ = [
    "ARKTTS_CODEC_FRAME_SIZE",
    "ARKTTS_CODEC_SAMPLE_RATE",
    "ARKTTS_RESIDUAL_CODEBOOK_SIZE",
    "ARKTTS_SEMANTIC_CODEBOOK_SIZE",
    "Audio8TTSConfig",
    "Audio8TTSFastARConfig",
    "Audio8TTSHybridSlowARConfig",
    "Audio8TTSSlowARConfig",
]
